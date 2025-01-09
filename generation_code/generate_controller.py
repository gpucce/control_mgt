import json
import pandas as pd

from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification 

from ctrl_llama import ControlledModel, get_ngram_mask
from utils import get_random_prompt_xsum, postprocess_text


def get_detectors(device):
    mage_tokenizer = AutoTokenizer.from_pretrained("yaful/MAGE")
    mage_clf = AutoModelForSequenceClassification.from_pretrained("yaful/MAGE").to(device)


    radar_tokenizer = AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
    radar_clf = AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B").to(device)

    return mage_clf, mage_tokenizer, radar_clf, radar_tokenizer


def prepare_inputs(tokenizer, prompts, add_gen_prompt=True, remove_eot=False):
    prompts = tokenizer.apply_chat_template(
        prompts, truncation=None, padding=False,
        add_generation_prompt=add_gen_prompt, tokenize=False)
    if remove_eot:
        prompts = [p.rstrip("<|eot_id|>") for p in prompts]
    prompts = tokenizer(prompts, return_tensors="pt")
    return prompts


def get_data(args, lines=True):
    data = pd.read_json(args.datapath, lines=lines)
    split = json.load(open("data/data_2024_11_08/splits/llama-3.1-70b-instruct-hf_xsum_informed.split.100000.json"))

    data_test = data[data.id.isin(split["te"])]
    data_train = data[data.id.isin(split["tr"])]
    
    return data_train, data_test


def get_models(models, device="cuda", training_docs=None, adapter_path="models-dpo/llama-3.1-8b_lora/control/2024-12-16-19-12"):
    model_dict = {}

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    if models == "all" or models == "llama":
        model_dict["llama"] = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct").to(device)
    if models == "all" or models == "dpo-llama":
        _adapter_name = "lora-dpo-adapter"
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct").to(device)
        model.load_adapter(adapter_path, adapter_name=_adapter_name) 
        model_dict["dpo-llama"] = model
    if models == "all" or models == "ctrl-llama":
        _num_controls = 1000
        model = ControlledModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct").to(device)
        bigram_mask, _ = get_ngram_mask(training_docs.sample(_num_controls, random_state=42).to_list(), tokenizer, model.vocab_size, n=2, verbose=True)
        model.set_logits_mask(bigram_mask.to_sparse())
    
    return model_dict, tokenizer


def main(args):
    data_train, data_test = get_data(args, lines=True)
    models, tokenizer = get_models(args.gen_model, device=args.device, training_docs=data_train)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_fn = f"test-xsum-newgen-{timestamp}.json"

    output_data = []
    for i, row in tqdm(data_test.iterrows(), total=data_test.shape[0]):
        data_row = {}

        prompt = get_random_prompt_xsum(m=row["title"], informed=True)
        model_inputs = prepare_inputs(tokenizer, prompt, add_gen_prompt=True, remove_eot=False)

        data_row["doc-id"] = row["id"]
        data_row["title"] = row["title"]

        data_row["human"] = row["real_article"]
        data_row["llama"] = row["generated_text"]

        for model_name, model in models.items():
            generated_ids = model.generate(**model_inputs.to(args.device), max_new_tokens=256, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            generated_text = postprocess_text(generated_text.split(".assistant")[-1]).lstrip()
            
            data_row[model_name] = generated_text
        
        output_data.append(data_row)

        with open(output_fn, "w") as jf:
            json.dump(output_data, jf, ensure_ascii=False)    


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, default="data/data_2024_11_12/generation_output_llama-3.1-8b-instruct-hf_xsum_temp0.8_informed_cut256.zip")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--gen_model", type=str, choices=["all", "ctrl-llama", "llama", "dpo-llama"], default="dpo-llama")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args)