import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from datetime import datetime
import random
import torch
import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model

from trl import DPOTrainer, DPOConfig
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from accelerate import Accelerator

import os
os.environ["WANDB_PROJECT"] = "control_mgt-dpo"


# Model Training Stalls with FSDP when fsdp_use_orig_params=False due to inconsistent model-optimizer state https://github.com/huggingface/accelerate/issues/3256
def get_dataset(dataset_name):
    if dataset_name == "control":
        _dataset = load_dataset("json", data_files="dpo_dataset/data/dataset-max-feature-difference-top-10_iter_1.zip", split="train")
        dataset = _dataset.train_test_split(test_size=0.2)
    elif dataset_name == "control-small":
        _dataset = load_dataset("json", data_files="dpo_dataset/data/control.small.jsonl", split="train")
        dataset = _dataset.train_test_split(test_size=0.2)
    else:
        raise NotImplementedError

    print(f"- data: {dataset_name} (len: {len(dataset)})")
    return dataset


def get_model(model_name, attn_impl, fsdp=None, lora_config=None, device="cuda"):
    if model_name == "llama-3.2":
        _model_name = "meta-llama/Llama-3.2-3B-Instruct"
    elif model_name == "llama-3.1-8b":
        _model_name = "meta-llama/Llama-3.1-8B-Instruct"
    else:
        raise NotImplementedError

    model = AutoModelForCausalLM.from_pretrained(
        _model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(_model_name,
                                              add_eos_token=True,
                                              add_bos_token=True,
                                              use_fast=False,
                                              )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    if lora_config is not None:
        print("- applying LoRA")
        # model = get_peft_model(model, lora_config, adapter_name="training_adapter")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    print(f"- model: {_model_name} (attn_impl: {attn_impl})")
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


def get_lora_config():
    config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
        )
    return config

def main(args):
    _start_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    accelerator = Accelerator()
    device = accelerator.device

    if args.lora:
        lora_config = get_lora_config()
    else:
        lora_config = None

    dataset = get_dataset(args.dataset)
    model, tokenizer = get_model(args.model_name, attn_impl=args.attn_impl, lora_config=lora_config, device=device)

    outdir = get_output_dir(args, start_time=_start_time, basedir="checkpoints-dpo")

    training_config = DPOConfig(
        output_dir=outdir,

        # training params
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
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
        beta=args.beta,
        eval_strategy="steps",
        save_strategy="steps" if not args.nosave else "no",
        logging_steps=10,
        save_total_limit=2,
        # report_to="wandb" if args.wandb else "none"
        run_name=f"{args.model_name}-dpo-barra-{_start_time}",

        # experimental
        use_liger_kernel=True,
        precompute_ref_log_probs=args.precompute_ref,
    )

    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_config
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
    parser.add_argument("--beta",               type=float, default=0.1)
    parser.add_argument("--lr"  ,               type=float, default=5e-7)
    parser.add_argument("--batch",              type=int,   default=1)
    parser.add_argument("--grad_acc",           type=int,   default=1)
    parser.add_argument("--eval_steps",         type=int,   default=500)
    parser.add_argument("--nepochs",            type=int,   default=1)
    parser.add_argument("--max_length",         type=int,   default=256)
    parser.add_argument("--warmup_ratio",       type=float, default=0.2)
    parser.add_argument("--attn_impl",          type=str,   default="sdpa", choices=["eager", "flash_attention_2", "sdpa"])
    parser.add_argument("--dataset",            type=str,   default="control")
    parser.add_argument("--fsdp",               action="store_true")
    parser.add_argument("--nosave",             action="store_true")
    parser.add_argument("--precompute_ref",     action="store_true")

    args = parser.parse_args()
    main(args)