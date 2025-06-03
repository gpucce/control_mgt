import os
import re
import json
import random
import joblib

import pandas as pd
import numpy as np

from transformers import AutoTokenizer

from pprint import pprint as pp
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

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



def create_dpo_dataset(selected_ids, hwt_text, mgt_text, mgt_method):
    dataset = []

    for doc_id in selected_ids:

        hwt_instance = hwt_text[hwt_text["doc-id"] == doc_id]
        mgt_instance = mgt_text[mgt_text["doc-id"] == doc_id]

        json_line = {'prompt': [], 'chosen': [], 'rejected': []}

        prompt = hwt_instance.prompt.item()
        json_line['prompt'].append({"role": "user", "content": prompt})

        # naive dataset: setting chosen content as the human-written text, rejectd otherwise
        json_line["chosen"].append({"role": "assistant", "content": hwt_instance["human"].item()})
        json_line["rejected"].append({"role": "assistant", "content": mgt_instance[mgt_method].item()})

        dataset.append(json_line)
    
    return dataset
    

def fill_missing_feats(data, all_features):
    for feat in all_features:
        if feat not in data.columns:
            data[feat] = 0
    return data[all_features]


def convert_ids(data):
    data.identifier = data.identifier.str.replace(".conllu", "")
    data.identifier = data.identifier.astype("int")
    return data


def get_split(data_mgt, data_hwt, idx):
    data_mgt = data_mgt[data_mgt.identifier.isin(idx)]
    data_mgt["label"] = 1

    data_hwt = data_hwt[data_hwt.identifier.isin(idx)]
    data_hwt["label"] = 0

    X_feats = pd.concat((data_mgt, data_hwt), axis=0, ignore_index=True)
    X_feats = shuffle(X_feats)

    labels = X_feats["label"].values
    ids = X_feats["identifier"]
    X_feats = X_feats.drop(columns=["identifier", "label"]).values

    return X_feats, labels, ids



def get_clf(model_path=None, feature_processor="zscore"):
    if model_path is not None:
        print(f"- loading trained model from: {model_path}")
        model = joblib.load(model_path)
    else:
        if feature_processor == "zscore":
            model = make_pipeline(StandardScaler(), LinearSVC(random_state=42, max_iter=1000, dual=False))
        elif feature_processor == "minmax":
            model = make_pipeline(MinMaxScaler(), LinearSVC(random_state=42, max_iter=1000, dual=False, ))
        else:
            raise NotImplementedError
    return model



def main(args):
    print_info_creation(args)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    naive_ids_outdir = os.path.join("dpo_dataset", "adversarial_dataset", args.dataset_name, args.mgt_method)
    os.makedirs(naive_ids_outdir, exist_ok=True)

    split_ids = json.load(open(args.split_file))
    
    print("- Building Naive DPO dataset")
    tr_ids = split_ids["tr"]

    hwt_text = pd.read_json(args.hwt_text, lines=True)
    mgt_text = pd.read_json(args.mgt_text, lines=False)

    hwt_text = hwt_text[hwt_text["doc-id"].isin(tr_ids)]
    mgt_text = mgt_text[mgt_text["doc-id"].isin(tr_ids)]

    selected_tr_ids = random.sample(tr_ids, args.num_samples)
    dpo_dataset = create_dpo_dataset(selected_ids=selected_tr_ids, hwt_text=hwt_text, mgt_text=mgt_text, mgt_method=args.mgt_method)

    # TODO store selected tr ids
    selected_tr_ids_df = pd.DataFrame(data={"identifier": selected_tr_ids})
    selected_tr_ids_df.to_csv(os.path.join(naive_ids_outdir, "selected_ids.csv"), index=False)

    # storing args to outdir
    with open(os.path.join(naive_ids_outdir, "args.json"), "w") as jf:
        json.dump(args.__dict__, jf, indent=2)

    # TODO store as zip file
    adversarial_dpo_dataset_fn = os.path.join(naive_ids_outdir, "adversarial_dpo_dataset.json")
    print(f"- storing naive adversarial dataset in: {adversarial_dpo_dataset_fn}")
    with open(adversarial_dpo_dataset_fn, "w") as jf:
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
    parser.add_argument("--second_iter", action="store_true")
    parser.add_argument("--prev_ids", type=str, default=None)
    args = parser.parse_args()

    main(args)