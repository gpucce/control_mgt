# https://github.com/ahans30/Binoculars
import os
import json
import pandas as pd
import torch

from sklearn.metrics import classification_report
from tqdm import tqdm
from itertools import batched

from Binoculars.binoculars import Binoculars


def get_data(datapath):
    if datapath.endswith(".zip"):
        data = pd.read_json(datapath, lines=True).to_dict(orient="records")
    else:
        data = json.load(open(datapath))
    return data


def main(args):
    data = get_data(args.datapath)

    if args.split_path is not None:
        test_split = json.load(open(args.split_path))["te"]
        data = pd.DataFrame(data)
        data["doc-id"] = data["doc-id"].astype(int)
        data = data[data["doc-id"].isin(test_split)]
        data = data.to_dict(orient="records")
    
    if args.datapath.endswith(".zip"):
        output_dir = os.path.join("evaluation_code", "evaluations", args.datapath.split("/")[-1].replace(".zip", ""), "binoculars_detector", args.target)
    else:
        output_dir = os.path.join("evaluation_code", "evaluations", *args.datapath.split("/")[2:-1], "binoculars_detector", args.target)

    print(f"- Evaluation binoculars INFO" + "-" * 25)
    print(f"- storing results in: {output_dir}")
    print(f"- target: {args.target}")
    print(f"- num pairs: {len(data)}")
    print("-" * 45)

    hwt = [elem["human"] for elem in data] 
    hwt_ids = [elem["doc-id"] for elem in data]
    mgt = [elem[args.target] for elem in data]
    mgt_ids = [elem["doc-id"] for elem in data]
    
    texts = hwt + mgt
    all_ids = hwt_ids + mgt_ids
    labels = [0 for i in range(len(hwt))] + [1 for i in range(len(mgt))]

    binoculars = Binoculars(max_token_observed=args.max_length)
    
    soft_preds = []
    hard_preds = []
    for elem in tqdm(batched(texts, args.batchsize), total=len(texts) // args.batchsize):
        with torch.no_grad():
            soft_p = binoculars.compute_score(elem)
            hard_p = binoculars.predict(elem)
        soft_preds.extend(soft_p)
        hard_preds.extend(hard_p)

    binoculars_pred_mapper = {
        "Most likely AI-generated": 1,
        "Most likely human-generated": 0,
    }
    hard_preds = [binoculars_pred_mapper[p] for p in hard_preds]
    
    metrics = classification_report(y_true=labels, y_pred=hard_preds, output_dict=True)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "clf_metrics.json"), "w") as jf:
        json.dump(metrics, jf)
    
    df_preds = pd.DataFrame(data={"doc-id": all_ids, "y_pred": hard_preds, "y_true": labels, "score": soft_preds})
    df_preds.to_csv(os.path.join(output_dir, "clf_preds.csv"), index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, default="generation_code/generations/xsum-iter-2/llama-dpo-iter2/0204-1431/generations-testset-0204_1532.json")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target", type=str, default="llama-dpo-iter2")
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--split_path", type=str, default=None)
    args = parser.parse_args()
    main(args)