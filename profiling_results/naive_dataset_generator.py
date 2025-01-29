import os
import random
import json
import re

import numpy as np
import pandas as pd

RANDOM_SEED = 42

pd.options.mode.chained_assignment = None


def print_info_creation(args):
    print("Naive Dataset Creation INFO " + "-" * 25)
    print(f"- datset name: {args.dataset_name}")
    print(f"- MGT text data path: {args.mgt_text}")
    print(f"- HWT text data path: {args.hwt_text}")
    print(f"- MGT generation method: {args.mgt_method}")

    print(f"- num samples naive dataset: {args.num_samples}")

    print(f"- naive dataset (ids) outdir: profiling_results/adversarial_dataset/{args.dataset_name}-naive/selected_ids.csv")

    print("-" * 50)


def __extract_systems_and_prompt(instance):
    # Updated pattern to capture all three blocks: system, user, and assistant
    pattern = (
        r"<\|start_header_id\|>system<\|end_header_id\|>\s*(.*?)<\|eot_id\|>\s*"  # Capture system text
        r"<\|start_header_id\|>user<\|end_header_id\|>\s*(.*?)<\|eot_id\|>\s*"  # Capture user prompt
    )

    # Perform the search
    match = re.search(pattern, instance, re.DOTALL)

    if match:
        system_text = match.group(1).strip()
        user_prompt = match.group(2).strip()

        _leftover_string = """Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n"""
        system_text = system_text.replace(_leftover_string, "")

        # Print the extracted texts
        """print("system_text:")
        print(system_text)
        print("\nuser_prompt:")
        print(user_prompt)"""
        return system_text, user_prompt
    else:
        print("No match found.")
        raise Exception("No System Prompt found")



def create_dpo_dataset(selected_ids, hwt_text, mgt_text, mgt_method):
    dataset = []

    for doc_id in selected_ids:

        hwt_instance = hwt_text[hwt_text["doc-id"] == doc_id]
        mgt_instance = mgt_text[mgt_text["doc-id"] == doc_id]

        system, prompt = __extract_systems_and_prompt(hwt_instance.prompt.item())
        json_line = {'prompt': [], 'chosen': [], 'rejected': []}
        json_line['prompt'].append({"role": "system", "content": system})
        json_line['prompt'].append({"role": "user", "content": prompt})

        # naive dataset: setting chosen content as the human-written text, rejectd otherwise
        json_line["chosen"].append({"role": "assistant", "content": hwt_instance["human"].item()})
        json_line["rejected"].append({"role": "assistant", "content": mgt_instance[mgt_method].item()})

        dataset.append(json_line)
    
    return dataset
    

def main(args):
    print_info_creation(args)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    naive_ids_outdir = os.path.join("profiling_results", "adversarial_dataset", f"{args.dataset_name}")
    os.makedirs(naive_ids_outdir, exist_ok=True)

    split_ids = json.load(open(args.split_file))
    
    print("- Building Naive DPO dataset")
    tr_ids = split_ids["tr"]

    hwt_text = pd.read_json(args.hwt_text, lines=True)
    mgt_text = pd.read_json(args.mgt_text, lines=True)

    selected_tr_ids = random.sample(tr_ids, args.num_samples)
    dpo_dataset = create_dpo_dataset(selected_ids=selected_tr_ids, hwt_text=hwt_text, mgt_text=mgt_text, mgt_method=args.mgt_method)

    # TODO store as zip file
    adversarial_dpo_dataset = os.path.join(naive_ids_outdir, "adversarial_dpo_dataset.json")
    print(f"- storing naive adversarial dataset in: {adversarial_dpo_dataset}")
    with open(adversarial_dpo_dataset, "w") as jf:
        json.dump(dpo_dataset, jf, ensure_ascii=False)    


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="xsum")
    parser.add_argument("--split_file", type=str, default="data/splits/split.100000.json")
    parser.add_argument("--mgt_text", type=str)
    parser.add_argument("--hwt_text", type=str)
    parser.add_argument("--mgt_method", type=str, default="llama")
    parser.add_argument("--num_samples", type=int, default=7000)
    args = parser.parse_args()

    main(args)