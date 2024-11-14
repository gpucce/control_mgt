import re
import os
import sys
import json
import random
import datasets
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(__file__))

from utils import get_regesto_prompt

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
        choices=["llama-3.1-8b-instruct-hf", "llama-3.1-70b-instruct-hf", "llama-3.1-405b-instruct-hf", "anita_8b"])
    return parser.parse_args()

def generate(prompts, llm, params):
    return llm.generate(prompts, sampling_params=params)

def prepare_inputs(prompts, llm):
    tokenizer = llm.get_tokenizer()
    prompts = tokenizer.apply_chat_template(
        prompts, truncation=None, padding=False,
        add_generation_prompt=True)
    prompts = tokenizer.batch_decode(prompts)
    return prompts

def get_tp_and_pp_size(model_name):
    tensor_parallel_size = 1
    pipeline_parallel_size = 1
    if "70" in model_name:
        tensor_parallel_size = 4
    if "405" in model_name:
        tensor_parallel_size = 16
    return tensor_parallel_size, pipeline_parallel_size

def get_vllm_llm_and_params(model_name: str, tokenizer_name: str):

    if tokenizer_name != model_name:
        print("vllm will ignore the tokenizer_name and use the same as model_name")

    params = SamplingParams(
        max_tokens=2048,
        min_tokens=256,
        temperature=0.8,
        top_p=0.9,
        # top_k=args.top_k if not args.use_beam_search else -1,
        # repetition_penalty=1.05,
        # use_beam_search=args.use_beam_search,
        # best_of=3 if args.use_beam_search else 1,
        # skip_special_tokens=True,
    )

    # this is not the right way
    tensor_parallel_size, pipeline_parallel_size = get_tp_and_pp_size(model_name)
    llm = LLM(
        model=model_name,
        tokenizer_mode="slow",
        tensor_parallel_size=tensor_parallel_size,
        # pipeline_parallel_size=pipeline_parallel_size,
        max_model_len=8192,
        enforce_eager=True)

    return llm, params

def postprocess_text(text):
    try:
        text = re.sub(r"\s+", " ", text)
    except:
        pass

    return text

def preprocess_samples(x):
    for col in ["regesto", "testo esteso", "apparato"]:
        if x[col] is not None:
            x[col] = " ".join([i.replace("¬ ", "").replace("¬", "") for i in x[col]])
    return x

experiments = {
    "format": get_regesto_prompt,
    "backtranslate": get_backtranslation_regesto_prompt,
}

if __name__ == "__main__":

    import os
    # import sys

    args = parse_args()
    my_folder = "/leonardo_scratch/large/userexternal/gpuccett/"
    models_folder = os.path.join(my_folder, "models/hf_llama/")
    data_path = os.path.join(my_folder, "Repos/MGH_annotation/output/escriptorium_mgh_1.json")
    df = datasets.load_dataset("json", data_files=str(data_path))["train"]
    df = df.map(preprocess_samples)
    df = df.to_pandas()
    _model_name = args.model_name
    model_path = os.path.join(models_folder, _model_name)
    tok = AutoTokenizer.from_pretrained(model_path)

    n_articles = 10000
    messages = df["testo esteso"].values[:n_articles]
    ids = df["numero"].values[:n_articles]
    regesti = df["regesto"].values[:n_articles]
    apparati = df["apparato"].values[:n_articles]
    
    llm, params = get_vllm_llm_and_params(model_path, model_path)
    for experiment, prompt_fn in experiments.items():
        prompts = [get_regesto_prompt(m, messages.tolist(), regesti.tolist(), 2) for idx, m in enumerate(messages.tolist())]
        prompts = prepare_inputs(prompts, llm)
    
        output_text = generate(prompts, llm, params)
    
        sep = "-"*10
        prompts = []
        outputs = []
        count = 0
        outfile = f"generation_output_{_model_name}_regesto_{experiment}.jsonl"
        with open(outfile, "w") as jf:
            for output, testo, regesto, apparato in zip(output_text, messages, regesti, apparati):
                count += 1
                prompt = output.prompt
                generated_text = postprocess_text(output.outputs[0].text)
    
                to_dump = {
                    "prompt": prompt,
                    "regesto_sintetico": generated_text,
                    "regesto_originale": regesto,
                    "apparato": apparato,
                    "testo_esteso": testo,
                }
    
                print(to_dump)
                jf.write(json.dumps(to_dump) + "\n")
    