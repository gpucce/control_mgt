import os
os.environ["HF_TOKEN"] = os.getenv("MY_HF_TOKEN")

import json
import torch

import pandas as pd

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from itertools import batched
from datetime import datetime

from utils import get_random_prompt_xsum, postprocess_text


def prepare_inputs(prompts, llm):
    tokenizer = llm.get_tokenizer()
    prompts = tokenizer.apply_chat_template(
        prompts, truncation=None, padding=False,
        add_generation_prompt=True)
    prompts = tokenizer.batch_decode(prompts)
    return prompts


def get_data(args, lines=True):
    data = pd.read_json(args.datapath, lines=lines)
    split = json.load(open("data/splits/split.100000.json"))

    data_test = data[data.id.isin(split["te"])]
    data_val = data[data.id.isin(split["val"])]
    data_train = data[data.id.isin(split["tr"])]
    
    return data_train, data_val, data_test


def get_model(model_name, adapter_path=None, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, model_id="models-dpo/llama-3.1-8b_lora/control-iter1/2025-01-25-11-06")
        model = model.merge_and_unload()
    
    return model, tokenizer


def get_vllm_model(model_name, enable_lora=False, max_tokens=1024):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LLM(
        model=model_name,
        max_model_len=1024,
        enable_lora=enable_lora,
        max_lora_rank=64,
        dtype=torch.bfloat16)
    sample_params = SamplingParams(max_tokens=max_tokens, min_tokens=256)
    return model, tokenizer, sample_params


def main(args):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    model_name = args.model_name

    outdir = os.path.join("generation_code", "generations", *args.adapter_path.split("/")[-2:])
    os.makedirs(outdir, exist_ok=True)
    output_fn = os.path.join(outdir, f"{args.output}{('-' + args.note) if args.note != '' else ''}-{timestamp}.json")

    data_train, data_val, data_test = get_data(args, lines=True)

    if args.alldata:
        data = pd.concat((data_train, data_val, data_test))
    else:
        data = data_test
        if args.test_num_samples:
            print(f"- Testing Mode: max docs set to: {args.test_num_samples}")
            data = data.sample(args.test_num_samples, random_state=42)
    
    print("Generation Recap" + "-" * 50)
    print(f"- datapath: {args.datapath}")
    print(f"- output file: {output_fn}")
    print(f"- model name: {args.model_name}")
    print(f"- adapter path: {args.adapter_path}")
    print(f"- notes: {args.note}")
    print(f"- data shape: {data.shape}")
    print("-" * 75)
    
    ids = data.id
    real_articles = data.real_article
    llama_articles = data.generated_text
    model, tokenizer, sample_params = get_vllm_model(
        model_name=args.model,
        enable_lora=True if args.adapter_path is not None else False,
        max_tokens=args.max_tokens
        )

    messages = data.title.values
    prompt_func = get_random_prompt_xsum
    prompts = [prompt_func(m, informed=True) for m in messages]
    prompts = prepare_inputs(prompts, model)

    if args.adapter_path is not None:
        lora_request = LoRARequest("dpo-1st", 1, lora_path=args.adapter_path)

    output_data = []
    for batch in batched(zip(prompts, messages, real_articles, llama_articles, ids), args.batch):
        prompt_batch, message_batch, real_batch, llama_batch, id_batch = zip(*batch)

        output_batch = model.generate(
            prompt_batch,
            sampling_params=sample_params,
            lora_request=lora_request if args.adapter_path is not None else None
            ) 
        
        for output, message, prompt, real_article, llama_article, _id in zip(
            output_batch, message_batch, prompt_batch, real_batch, llama_batch, id_batch):
            prompt = output.prompt
            prompts.append(prompt)
            generated_text = postprocess_text(output.outputs[0].text)
            real_article = postprocess_text(real_article)

            to_dump = {
                "doc-id": str(_id),
                "title": message,
                "human": real_article,
                "llama": llama_article,
                model_name: generated_text,
                }
            output_data.append(to_dump)
                
        with open(output_fn, "w") as jf:
            json.dump(output_data, jf, ensure_ascii=False)    
 

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, default="data/xsum_generations/vanilla/xsum-generations.zip")
    parser.add_argument("--adapter_path", type=str, default="models-dpo/llama-3.1-8b_lora/adversarial-dpo-iter1-filtered/2025-01-28-18-49")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output", type=str, default="xsum")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--alldata", action="store_true")
    parser.add_argument("--test_num_samples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=1024)

    args = parser.parse_args()
    main(args)