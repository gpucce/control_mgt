import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from datetime import datetime
import random
import torch
import os

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM 

from trl import DPOTrainer, DPOConfig
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from accelerate import Accelerator

import os

# TODO check system prompt 

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"- trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {round(100 * trainable_params / all_param, 4)}"
    )


# Model Training Stalls with FSDP when fsdp_use_orig_params=False due to inconsistent model-optimizer state https://github.com/huggingface/accelerate/issues/3256
def get_dataset(dataset_name):
    if dataset_name == "adversarial-dpo-iter1":
        _dataset = load_dataset("json", data_files="profiling_results/adversarial_dataset/xsum/dpo-iter1/adversarial_dpo_dataset.json", split="train")
    elif dataset_name == "adversarial-dpo-iter1-filtered":
        _dataset = load_dataset("json", data_files="profiling_results/adversarial_dataset/xsum/dpo-iter1-filtered-cut256/adversarial_dpo_dataset.json", split="train")
    # elif dataset_name == "andrea-dpo-iter1-unfiltered":
    #     # _dataset = load_dataset("json", data_files="profiling_results/adversarial_dataset/xsum-wrong-feats/dpo-iter1-unfiltered/adversarial_dpo_dataset.json", split="train")
    #     _dataset = load_dataset("json", data_files="profiling_results/adversarial_dataset/xsum/dpo-iter1-unfiltered-cut256", split="train")
    elif dataset_name == "adversarial-dpo-iter2-filtered":
        # TODO
        raise NotImplementedError
    elif dataset_name == "adversarial-naive-dpo-iter1":
        _dataset = load_dataset("json", data_files="profiling_results/adversarial_dataset/xsum/dpo-iter1-naive-cut256/adversarial_dpo_dataset.json", split="train")
    elif dataset_name == "adversarial-dpo-iter1-filtered-zscore":
        _dataset = load_dataset("json", data_files="profiling_results/adversarial_dataset/xsum/dpo-iter1-filtered-cut256-zscore/adversarial_dpo_dataset.json", split="train")
    elif dataset_name == "adversarial-dpo-iter1-unfiltered-zscore":
        _dataset = load_dataset("json", data_files="profiling_results/adversarial_dataset/xsum/dpo-iter1-unfiltered-cut256-zscore/adversarial_dpo_dataset.json", split="train")
    else:
        raise NotImplementedError
    
    dataset = _dataset.train_test_split(test_size=0.2)
    print(f"- data: {dataset_name} (train-len: {len(dataset['train'])})")
    return dataset


def _get_model_name(model_name):
    if model_name == "llama-3.2":
        _model_name = "meta-llama/Llama-3.2-3B-Instruct"
    elif model_name == "llama-3.1-8b":
        _model_name = "meta-llama/Llama-3.1-8B-Instruct"
    elif model_name == "gemma":
        _model_name = "google/gemma-2-2b-it"
    else:
        raise NotImplementedError(f"'{model_name}' not valid!")
    return _model_name


def _get_model(model_name, pretrained_adapter=None, lora_config=None, fsdp=False, attn_impl="flash_attention_2", is_peft_trainable=False, device="cuda"):
    _model_name = _get_model_name(model_name)
    
    if pretrained_adapter is not None:
        print(f"- loading pre-trained LoRA from: {pretrained_adapter}")
        model = AutoPeftModelForCausalLM.from_pretrained(pretrained_adapter, is_trainable=is_peft_trainable, torch_dtype=torch.bfloat16, device_map="cpu", token=os.getenv("MY_HF_TOKEN")).to(device)
    else:
        if lora_config is None:
            model = AutoModelForCausalLM.from_pretrained(_model_name, torch_dtype=torch.bfloat16, device_map="cpu", token=os.getenv("MY_HF_TOKEN")).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(_model_name, torch_dtype=torch.bfloat16, device_map="cpu", token=os.getenv("MY_HF_TOKEN")).to(device)
            model = get_peft_model(model, peft_config=lora_config)
    
    tokenizer = AutoTokenizer.from_pretrained(_model_name, add_eos_token=True, add_bos_token=True, use_fast=False, token=os.getenv("MY_HF_TOKEN"))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    print(f"- model: {_model_name} (attn_impl: {attn_impl})")
    print_trainable_parameters(model)

    return model, tokenizer


def get_output_dir(args, start_time, basedir="checkpoints-dpo"):
    model = args.model_name
    dataset = args.dataset
    
    if args.lora:
        model += "_lora"
    
    outdir = os.path.join(basedir, model, dataset, start_time)
    os.makedirs(outdir, exist_ok=True)

    print(f"- output dir: {outdir}")
    return outdir


def get_lora_config(rank=32):
    config = LoraConfig(
        r=rank,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        )
    return config


def main(args):
    _start_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    accelerator = Accelerator()
    device = accelerator.device

    if args.lora:
        if args.adapter_path is not None:
            lora_config = {"adapter_path": args.adapter_path} 
        else:
            lora_config = get_lora_config(rank=args.lora_rank)
    else:
        lora_config = None

    dataset = get_dataset(args.dataset)
    model, tokenizer = _get_model(args.model_name, pretrained_adapter=args.adapter_path, attn_impl=args.attn_impl, lora_config=lora_config, device=device, is_peft_trainable=True)
    earlystop = EarlyStoppingCallback(early_stopping_patience=10)

    outdir = get_output_dir(args, start_time=_start_time, basedir="checkpoints-dpo")
    print(f"- outdir: {outdir}")

    training_config = DPOConfig(
        output_dir=outdir,

        # training params
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 4,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        bf16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        num_train_epochs=args.nepochs,
        max_length=args.max_length,

        # Run params
        metric_for_best_model="eval_rewards/margins",
        greater_is_better=True,
        load_best_model_at_end=True,
        logging_first_step=True,
        eval_on_start=True,
        beta=args.beta,
        eval_strategy="steps",
        save_strategy="steps" if not args.nosave else "no",
        logging_steps=10,
        save_total_limit=2,
        run_name=f"{args.model_name}-{args.dataset}-{_start_time}",

        # experimental
        # use_liger_kernel=True,
        precompute_ref_log_probs=args.precompute_ref,
    )

    # Set reference model for >=2 dpo iterations
    if "iter2" in args.dataset:
        print("- loading LoRA reference model")
        ref_model, _ = _get_model(args.model_name, pretrained_adapter=args.adapter_path, attn_impl=args.attn_impl, device=device, is_peft_trainable=False)
    else:
        ref_model = None

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_config,
        callbacks=[earlystop],
    )
    
    trainer.train()

    if args.nosave:
        os.rmdir(outdir)
    
    if args.fsdp:
        print("- unwrapping FSDP model")
        model = accelerator.unwrap_model(model)

    # FIXME merging lora weights when FSDP causes mismatch error -> atm we're simply saving the adapters
    # if args.lora:
    #     print("- merging peft adapters into model")
    #     # model = model.merge_and_unload(adapter_names=["training_adapter"])
    #     model = model.merge_and_unload()
    
    outdir_final = get_output_dir(args, start_time=_start_time, basedir="models-dpo")

    if args.fsdp:
        print("- saving final model (via accelerate save)")
        model.save_pretrained(
            outdir_final,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
    else:
        print("- saving final model")
        model.save_pretrained(
            outdir_final,
            is_main_process=accelerator.is_main_process,
        )
    

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_name",         type=str,   default="llama-3.2")
    parser.add_argument("--lora",               action="store_true")
    parser.add_argument("--adapter_path",       type=str,   default=None, help="path to pre-trained LoRA module")
    parser.add_argument("--beta",               type=float, default=0.1)
    parser.add_argument("--lr"  ,               type=float, default=5e-7)
    parser.add_argument("--batch",              type=int,   default=1)
    parser.add_argument("--grad_acc",           type=int,   default=1)
    parser.add_argument("--eval_steps",         type=int,   default=500)
    parser.add_argument("--nepochs",            type=int,   default=1)
    parser.add_argument("--max_length",         type=int,   default=256)
    parser.add_argument("--warmup_ratio",       type=float, default=0.2)
    parser.add_argument("--attn_impl",          type=str,   default="sdpa", choices=["eager", "flash_attention_2", "sdpa"])
    parser.add_argument("--dataset",            type=str,   default="adversarial-dpo-iter1-filtered")
    parser.add_argument("--fsdp",               action="store_true")
    parser.add_argument("--nosave",             action="store_true")
    parser.add_argument("--precompute_ref",     action="store_true")
    parser.add_argument("--wandb_project",      type=str, default="control_mgt-dpo")
    parser.add_argument("--lora_rank",          type=int, default=32)

    args = parser.parse_args()
    os.environ["WANDB_PROJECT"] = args.wandb_project
    main(args)