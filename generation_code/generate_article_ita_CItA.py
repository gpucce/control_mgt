import re
import os
import sys
import json
import random
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(__file__))

from utils import get_random_prompt_cita, get_random_prompt_cita_anita

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
        choices=[
            "llama-3.1-8b-instruct-hf", "llama-3.1-70b-instruct-hf", "llama-3.1-405b-instruct-hf",
            "llama-3-70b-instruct-hf", "anita_8b"])
    return parser.parse_args()

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
    distributed_executor_backend = "mp"
    if "70" in model_name:
        tensor_parallel_size = 4
    if "405" in model_name:
        tensor_parallel_size = 16
        pipeline_parallel_size = 1
        distributed_executor_backend = "ray"
    return {
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "distributed_executor_backend":distributed_executor_backend
    }

def get_vllm_llm_and_params(model_name: str, tokenizer_name: str):

    if tokenizer_name != model_name:
        print("vllm will ignore the tokenizer_name and use the same as model_name")

    params = SamplingParams(
        max_tokens=2048,
        min_tokens=512,
        # POSSIBLE PARAMS
        temperature=0.3,
        top_p=0.9,
        # top_k=args.top_k if not args.use_beam_search else -1,
        # repetition_penalty=1.05,
        # use_beam_search=args.use_beam_search,
        # best_of=3 if args.use_beam_search else 1,
        # skip_special_tokens=True,
    )

    # this is not the right way
    vllm_kwargs = get_tp_and_pp_size(model_name)
    llm = LLM(
        model=model_name,
        tokenizer_mode="slow",
        max_model_len=2048,
        enforce_eager=True,
        **vllm_kwargs)

    return llm, params

def generate(prompts, llm, params):
    return llm.generate(prompts, sampling_params=params)

def postprocess_text(text):
    try:
        text = re.sub(r"\s+", " ", text)
    except:
        pass

    return text

if __name__ == "__main__":

    import os
    # import sys

    args = parse_args()
    my_folder = "/leonardo_scratch/large/userexternal/gpuccett/"
    models_folder = os.path.join(my_folder, "models/hf_llama/")
    data_path = os.path.join(my_folder, "Repos/MGT2025-private/generate_article_ita_news/cita.csv")
    df = pd.read_csv(data_path)
    df = df.loc[df.text.apply(lambda x: 1000 < len(x) < 3000 if isinstance(x, str) else False), :]
    _model_name = args.model_name
    model_path = os.path.join(models_folder, _model_name)
    tok = AutoTokenizer.from_pretrained(model_path)

    n_articles = 1000
    messages = df["title"].values[:n_articles]
    ids = df["id"].values[:n_articles]
    real_essays = df["text"].values[:n_articles]
    prompt_func = get_random_prompt_cita if "anita" not in _model_name else get_random_prompt_cita_anita
    prompts = [prompt_func(m) for m in messages]

    llm, params = get_vllm_llm_and_params(model_path, model_path)
    prompts = prepare_inputs(prompts, llm)

    # output_text = []
    # for i in prompts:
    #     output_text += generate([i], llm, params)
    output_text = generate(prompts, llm, params)

    sep = "-" * 10
    prompts = []
    outputs = []
    count = 0
    with open(f"generation_output_{_model_name}_cita.jsonl", "w") as jf:
        for output, message, real_essay, _id in zip(output_text, messages, real_essays, ids):
            count += 1
            prompt = output.prompt
            prompts.append(prompt)
            generated_text = postprocess_text(output.outputs[0].text)
            real_essay = postprocess_text(real_essay)

            to_dump = {
                "prompt": prompt,
                "generated_text": generated_text,
                "real_essay": real_essay,
                "id": str(_id),
                "source": "cita",
            }

            jf.write(json.dumps(to_dump) + "\n")
