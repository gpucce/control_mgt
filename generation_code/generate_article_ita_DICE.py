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

from utils import get_random_prompt_dice, get_random_prompt_dice_anita

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
        choices=["llama-3.1-8b-instruct-hf", "llama-3.1-70b-instruct-hf", "llama-3.1-405b-instruct-hf", "anita_8b"])
    parser.add_argument("--informed", action="store_true", default=False)
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
    tensor_parallel_size, pipeline_parallel_size = get_tp_and_pp_size(model_name)
    llm = LLM(
        model=model_name,
        tokenizer_mode="slow",
        tensor_parallel_size=tensor_parallel_size,
        # pipeline_parallel_size=pipeline_parallel_size,
        max_model_len=4096,
        enforce_eager=True)

    return llm, params

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
    data_path = os.path.join(my_folder, "datasets/Italian-Crime-News/italian_crime_news.csv")
    df = pd.read_csv(data_path, encoding="latin")
    df = df.loc[df.text.apply(lambda x: len(x) > 1000 if isinstance(x, str) else False), :]
    _model_name = args.model_name
    model_path = os.path.join(models_folder, _model_name)
    tok = AutoTokenizer.from_pretrained(model_path)

    n_articles = 10000
    messages = df["title"].values[:n_articles]
    ids = df["id"].values[:n_articles]
    crime_tags = df["newspaper_tag"].values[:n_articles]
    real_articles = df["text"].values[:n_articles]
    prompt_func = get_random_prompt_dice if "anita" not in _model_name else get_random_prompt_dice_anita
    prompts = [prompt_func(m, args.informed) for m in messages]

    llm, params = get_vllm_llm_and_params(model_path, model_path)
    prompts = prepare_inputs(prompts, llm)

    # output_text = []
    # for i in prompts:
    #     output_text += generate([i], llm, params)

    output_text = generate(prompts, llm, params)

    sep = "-"*10
    prompts = []
    outputs = []
    count = 0
    outfile = f"generation_output_{_model_name}_dice.jsonl"
    if args.informed:
        outfile = outfile.replace(".jsonl", "_informed.jsonl")
    with open(outfile, "w") as jf:
        for output, message, real_article, _id, crime_tag in zip(output_text, messages, real_articles, ids, crime_tags):
            count += 1
            prompt = output.prompt
            prompts.append(prompt)
            generated_text = postprocess_text(output.outputs[0].text)
            print(sep, f"Prompt: {prompt}")
            print(sep, f"Generated text: {generated_text}")
            real_article = postprocess_text(real_article)
            print(sep, real_article)

            to_dump = {
                "prompt": prompt,
                "generated_text": generated_text,
                "real_article": real_article,
                "crime_tag": crime_tag,
                "id": str(_id),
                "title": message,
                "source": "DICE_ita",
            }

            print(to_dump)
            jf.write(json.dumps(to_dump) + "\n")
