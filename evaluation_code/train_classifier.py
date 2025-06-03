import warnings
warnings.simplefilter(action='ignore')
print("- NB: suppressing warnings!")

import os
import wandb
import json
import pandas as pd
import numpy as np
import torch

from collections import Counter
from argparse import ArgumentParser
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel, FlaxAutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from peft import PeftModel

import os
os.environ["WANDB_PROJECT"] = "control_mgt"


class NewsDataset(Dataset):
    def __init__(
            self,
            data,
            skip_nchars=0,
            lowercase=False,
    ):
        self.data = data
        self.skip_nchars = skip_nchars
        self.lowercase = lowercase
    
    def __getitem__(self, index):
        data = self.data.iloc[index]
        sample = {
            "id": data.id,
            "text": data.text[self.skip_nchars:],
            "label": data.label
        }

        if self.lowercase:
            sample["text"] = sample["text"].lower()

        return sample
    
    def __len__(self):
        return len(self.data)


class CustomCollate:
    def __init__(
            self,
            tokenizer,
            max_length
    ):
       self.tokenizer = tokenizer
       self.max_length = max_length
    
    def __call__(self, batch):
        texts = [elem["text"] for elem in batch]
        labels = [elem["label"] for elem in batch]
        model_inputs = self.tokenizer(texts, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        model_inputs["labels"] = torch.tensor(labels)
        return model_inputs 


class LLM2VecCustomCollate:
    def __init__(
        self,
        tokenizer,
        max_length=512,
        doc_max_length=412
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_max_length = doc_max_length

    def __call__(self, batch):
        texts = [elem["text"] for elem in batch]
        labels = [elem["label"] for elem in batch]
        model_inputs = llm2vec_tokenize(self.tokenizer, texts)
        model_inputs["labels"] = torch.tensor(labels)
        return model_inputs


def custom_compute_metrics(preds):
    _preds = preds.predictions.argmax(axis=1)
    _labels = preds.label_ids
    acc = accuracy_score(y_true=_labels, y_pred=_preds)
    recall = recall_score(y_true=_labels, y_pred=_preds)
    precision = precision_score(y_true=_labels, y_pred=_preds)
    f1 = f1_score(y_true=_labels, y_pred=_preds, average="micro")
    return {"acc": acc, "recall": recall, "precision": precision, "f1": f1}


def get_model(model_name, device="cuda"):
    if model_name == "deberta":
        model_id = "microsoft/deberta-v3-base"
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    elif model_name == "roberta":
        model_id = "FacebookAI/roberta-base"
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    elif model_name == "llm2vec-llama":
        base_model_id = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp"
        model_id = "McGill-NLP/LLM2Vec-Meta-Llama-31-8B-Instruct-mntp-supervised"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        # Loading base model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
        config = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(base_model_id, trust_remote_code=True, config=config, torch_dtype=torch.bfloat16, device_map="cuda" if device == "cuda" else "cpu")
        model = PeftModel.from_pretrained(model, base_model_id)
        model = model.merge_and_unload()
        # Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
        model = PeftModel.from_pretrained(model, model_id)
        model = LLM2VecForSequenceClassification(base_model=model, num_labels=2, tokenizer=tokenizer).to(device)
    elif "checkpoints-classifier" in model_name:
        model_id = model_name
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise NotImplementedError(f"model: {model_name} not implemented yet!")
    print(f"- Loaded model: {model_id}")
    return model, tokenizer


def compute_tokenized_lengths(data, tokenizer):
    real    = [len(elem) for elem in tokenizer([sample["r_article"] for sample in data])["input_ids"]]
    synth   = [len(elem) for elem in tokenizer([sample["g_article"] for sample in data])["input_ids"]]

    r_std, r_mean, r_max, r_min = np.std(real), np.mean(real), np.max(real), np.min(real)
    f_std, f_mean, f_max, f_min = np.std(synth), np.mean(synth), np.max(synth), np.min(synth)
    return (r_mean, r_std, r_max, r_min), (f_mean, f_std, f_max, f_min)
    

def create_dataset(df_data, only_synth=False):
    df_real         = df_data[["real_article", "id", "source"]].rename(columns={"real_article": "text"})
    # df_real = df_real.iloc[:len(df_real) // 2, :]
    df_generated    = df_data[["generated_text", "id", "source"]].rename(columns={"generated_text": "text"})
    # df_generated = df_generated.iloc[len(df_generated) // 2:, :]
    
    df_real["label"] = 0
    df_generated["label"] = 1

    if not only_synth:
        df_data = pd.concat([df_real, df_generated], ignore_index=True)
    else:
        df_data = df_generated

    return df_data


def main(args):
    if args.eval_only:
        fast_eval_only(args)
        exit()

    data = pd.read_json(args.datapath, lines=True)
    selected_fname = "data/splits/split." + str(args.num_samples) + ".json"
    selected = json.load(open(selected_fname))

    tr_data = data[data["id"].isin(selected["tr"])]
    va_data = data[data["id"].isin(selected["val"])]
    te_data = data[data["id"].isin(selected["te"])]

    if args.test_othergen:
        selected_othergen_fname = selected_fname.replace("llama-3.1-8b", "llama-3.1-70b") if "llama3.1-8b" in selected_fname else selected_fname.replace("llama-3.1-70b", "llama-3.1-8b")
        datapath_othergen = args.datapath.replace("llama-3.1-70b", "llama-3.1-8b") if "llama-3.1-70b" in args.datapath else args.datapath.replace("llama-3.1-8b", "llama-3.1-70b")
        data_other_gen = pd.read_json(datapath_othergen, lines=True)
        selected_other_gen = json.load(open(selected_othergen_fname))
        te_data_other_gen = data_other_gen[data_other_gen["id"].isin(selected_other_gen["te"])]
        te_data_other_gen = create_dataset(te_data_other_gen)

    # if args.num_samples is not None:
    #     selected_fname = args.datapath.rstrip(".jsonl").replace("data/generation_output_", "data/splits") + f".split.{args.num_samples}.json"
    #     selected = json.load(open(selected_fname))
    #     
    #     print(f"Setting max number of document pairs to: {args.num_samples}")
    #     data = data[:args.num_samples]

    # tr_data, va_data = train_test_split(data,       train_size=0.70, random_state=42)
    # va_data, te_data = train_test_split(va_data,    train_size=0.25, random_state=42)

    tr_data = create_dataset(tr_data)
    va_data = create_dataset(va_data)
    te_data = create_dataset(te_data)

    if args.wandb:
        import wandb
        run_name=f"{args.model_name}" if not args.freeze_backbone else f"{args.model_name}.frozen"
        wandb.init(name=run_name, 
                   config={
                       "num_samples": args.num_samples,
                       "data_max_length": args.max_length,
                       "skip_nchars": args.skip_nchars,
                       "freeze_backbone": args.freeze_backbone,
                       "model_name": args.model_name,
                       "dataset": args.datapath.split("/")[-1]
                       })

    print(f"\nDataset Statistics:\n- Training Data documents:\t{len(tr_data)}\t({Counter(tr_data.label)})\n- Validation Data documents:\t{len(va_data)}\t({Counter(va_data.label)})\n- Testing Data documents:\t{len(te_data)}\t({Counter(te_data.label)})\n")

    tr_dataset = NewsDataset(tr_data, skip_nchars=args.skip_nchars)
    va_dataset = NewsDataset(va_data, skip_nchars=args.skip_nchars)
    te_dataset = NewsDataset(te_data, skip_nchars=args.skip_nchars)
    
    if args.test_othergen:
        te_dataset_other_gen = NewsDataset(te_data_other_gen, skip_nchars=args.skip_nchars)

    # splits = {}
    # for (_data, _name) in [(tr_data, "tr"), (va_data, "val"), (te_data, "te")]:
    #     os.makedirs("data/splits", exist_ok=True)
    #     nsamples = args.num_samples if args.num_samples is not None else "100k"
    #     _data = _data.id.drop_duplicates()
    #     splits[_name] = _data.to_list()
    
    # with open(f"{args.datapath.replace('generation_output_', f'splits/').replace('jsonl', f'split.{nsamples}.json')}", "w") as f:
    #     json.dump(splits, f)

    model, tokenizer = get_model(model_name=args.model_name, device="cuda")

    if args.drop_positions:
        print("- freezing positional embeddings (zero-like)")
        model.base_model.embeddings.position_embeddings.weight.data = torch.zeros_like(model.base_model.embeddings.position_embeddings.weight)
        model.base_model.embeddings.position_embeddings.weight.requires_grad = False


    if args.stats:
        print("\nTraining Tokenized Stats: ")
        tr_r_stats, tr_f_stats = compute_tokenized_lengths(tr_dataset, tokenizer)
        print(f"Tokenized Real Texts Statistics - mean: {tr_r_stats[0]:.2f} (std: {tr_r_stats[1]:.2f}, max: {tr_r_stats[2]}, min: {tr_r_stats[3]})")
        print(f"Tokenized Fake Texts Statistics - mean: {tr_f_stats[0]:.2f} (std: {tr_f_stats[1]:.2f}, max: {tr_f_stats[2]}, min: {tr_f_stats[3]})")

        print("Validation Tokenized Stats:")
        va_r_stats, va_f_stats = compute_tokenized_lengths(va_dataset, tokenizer)
        print(f"Tokenized Real Texts Statistics - mean: {va_r_stats[0]:.2f} (std: {va_r_stats[1]:.2f}, max: {va_r_stats[2]}, min: {va_r_stats[3]})")
        print(f"Tokenized Fake Texts Statistics - mean: {va_f_stats[0]:.2f} (std: {va_f_stats[1]:.2f}, max: {va_f_stats[2]}, min: {va_f_stats[3]})")
        
        print("Test Tokenized Stats: ")
        te_r_stats, te_f_stats = compute_tokenized_lengths(te_dataset, tokenizer)
        print(f"Tokenized Real Texts Statistics - mean: {te_r_stats[0]:.2f} (std: {te_r_stats[1]:.2f}, max: {te_r_stats[2]}, min: {te_r_stats[3]})")
        print(f"Tokenized Fake Texts Statistics - mean: {te_f_stats[0]:.2f} (std: {te_f_stats[1]:.2f}, max: {te_f_stats[2]}, min: {te_f_stats[3]})\n")

    if args.freeze_backbone:
        print("\n- Freezing backbone model")
        if "llm2vec-" in args.model_name:
            for _, p in model.model.named_parameters():
                p.requires_grad = False
        else:
            for _, p in model.base_model.named_parameters():
                p.requires_grad = False 

        trainable_layers = [p_name for p_name, p in model.named_parameters() if p.requires_grad == True] 
        print(f"Trainable layers: {trainable_layers}")

    # output_folder = f"{args.model_name}_split{args.num_samples}_skip{args.skip_nchars}/{args.datapath}" if not args.freeze_backbone else f"{args.model_name}_split{args.num_samples}_frozen/{args.datapath}" 

    def get_output_folder(model_name, num_samples, skip_nchars, datapath, no_positions, freeze_backbone):
        no_positions = "nopos" if no_positions else ""
        skip_nchars = f"skip_{skip_nchars}" if skip_nchars != 0 else ""
        freeze_backbone = "frozen" if freeze_backbone else ""
        num_samples = f"split{num_samples}"
        datapath = datapath.split("/")[-1].replace(".zip", "")

        path = [elem for elem in [model_name, num_samples, no_positions, freeze_backbone, skip_nchars, datapath] if elem != ""]

        output_folder = os.path.join(*path)
        return output_folder
    
    output_folder = get_output_folder(model_name=args.model_name, num_samples=args.num_samples, skip_nchars=args.skip_nchars, datapath=args.datapath, no_positions=args.drop_positions, freeze_backbone=args.freeze_backbone)

    trainer_args = TrainingArguments(
        output_dir=f"checkpoints-classifier/{output_folder}",
        bf16=True,
        learning_rate=args.lr,
        num_train_epochs=args.nepochs,
        logging_steps=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=25,
        save_steps=25,
        weight_decay=0.0,
        run_name=f"{args.model_name}" if not args.freeze_backbone else f"{args.model_name}.frozen",
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb" if args.wandb else "none",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        save_total_limit=1,
        eval_on_start=True,
    )

    callbacks = []
    if args.patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))

    if "llm2vec-" in args.model_name:
        collate_fn = LLM2VecCustomCollate(tokenizer, max_length=args.max_length, doc_max_length=412)
    else:
        collate_fn = CustomCollate(tokenizer, max_length=args.max_length)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        train_dataset=tr_dataset,
        eval_dataset=va_dataset,
        compute_metrics=custom_compute_metrics,
        args=trainer_args,
        callbacks=callbacks
    )

    print("\nTraining:")
    trainer.train()

    test_results = trainer.predict(test_dataset=te_dataset)
    
    if args.test_othergen:
        test_results_other_generator = trainer.predict(test_dataset=te_dataset_other_gen, metric_key_prefix="test_othergen")
    
    from pprint import pprint as pp
    print("\nTest Results:")
    pp({k: round(v, 4) for k, v in test_results.metrics.items()})

    if args.test_othergen:
        print("\nTest Other Generator Results:")
        pp({k: round(v, 4) for k, v in test_results_other_generator.metrics.items()})

    if args.wandb:
        wandb.finish()
    

def fast_eval_only(args):
    from transformers.utils.logging import disable_progress_bar
    disable_progress_bar()

    data = pd.read_json(args.datapath, lines=True)
    selected_fname = "data/splits/split." + str(args.num_samples) + ".json"
    selected = json.load(open(selected_fname))

    te_data = data[data["id"].isin(selected["te"])]
    te_data = create_dataset(te_data, only_synth=args.only_synth)

    te_dataset = NewsDataset(te_data, skip_nchars=args.skip_nchars)
    print(f"\nDataset Statistics:\n- Testing Data documents:\t{len(te_data)}\t({Counter(te_data.label)})\n")
    
    model, tokenizer = get_model(model_name=args.model_name, device="cuda")

    preds_fname = get_preds_fname(args.model_name, args.datapath, args.max_length, args.only_synth, args.skip_nchars)
    output_folder = f"{args.model_name}/{args.datapath}" if not args.freeze_backbone else f"{args.model_name}_frozen/{args.datapath}" 
    trainer_args = TrainingArguments(
        output_dir=f"eval_logs/{output_folder}",
        bf16=True,
        learning_rate=args.lr,
        num_train_epochs=args.nepochs,
        logging_steps=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=25,
        save_steps=25,
        weight_decay=0.0,
        run_name=f"{args.model_name}" if not args.freeze_backbone else f"{args.model_name}.frozen",
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="wandb" if args.wandb else "none",
        load_best_model_at_end=True,
        remove_unused_columns=False,
        save_total_limit=1,
        do_train=False,
    )

    if "llm2vec-" in args.model_name:
        collate_fn = LLM2VecCustomCollate(tokenizer, max_length=args.max_length, doc_max_length=412)
    else:
        collate_fn = CustomCollate(tokenizer, max_length=args.max_length)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=custom_compute_metrics,
        args=trainer_args,
    )


    test_results = trainer.predict(test_dataset=te_dataset)

    preds = test_results.predictions.argmax(axis=1)

    df_pred = pd.DataFrame(columns=["id", "pred", "label"])
    if args.only_synth:
        df_pred["id"] = selected["te"]
    else:
        df_pred["id"] = selected["te"] * 2
    df_pred["pred"] = preds
    df_pred["label"] = test_results.label_ids

    if not args.nosave:
        df_pred.to_csv(preds_fname, index=False)
    
    from pprint import pprint as pp
    print("\nTest Results:")
    pp({k: round(v, 4) for k, v in test_results.metrics.items()})

    import shutil
    shutil.rmtree("eval_logs")   # remove eval_logs dir created by trainer (it is an empty dir since we're only doing eval in here...)

    return


def get_preds_fname(model_name, datapath, max_length, only_synth, skip_nchars, basedir="preds-classifier"):
    if "checkpoints-classifier" in model_name:
        _modelname = model_name.split("/")
        _datapath = datapath.split("/")
        if skip_nchars != 0:
            preds_fname = os.path.join(basedir, _modelname[1], f"{_datapath[1]}_skip{skip_nchars}")
        else:
            preds_fname = os.path.join(basedir, _modelname[1], _datapath[1])
        os.makedirs(preds_fname, exist_ok=True)
        preds_fname = os.path.join(preds_fname, f"preds_{_modelname[-1]}.{_datapath[-1].replace('.zip', '')}.{max_length}{'.only_synth' if only_synth else ''}.csv")
    else:
        raise NotImplementedError
    return preds_fname 


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False, default="deberta")
    parser.add_argument("--datapath", type=str, required=False, default="data/data_2024_11_08/generation_output_llama-3.1-70b-instruct-hf_xsum_temp0.8_informed.zip")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--skip_nchars", type=int, default=0)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--test_othergen", action="store_true", help="test on other generator too")     # TODO
    parser.add_argument("--only_synth", action="store_true", help="evaluate only on synthetic texts")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--drop_positions", action="store_true")
    args = parser.parse_args()
    main(args)
