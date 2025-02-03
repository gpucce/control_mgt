import os
import json
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from tqdm import tqdm
from itertools import batched


def get_data(datapath):
    if datapath.endswith(".zip"):
        data = pd.read_json(datapath, lines=True).to_dict(orient="records")
    else:
        data = json.load(open(datapath))
    return data


def mage_detect(logits, th=-3.08583984375):
    # adapted from https://github.com/yafuly/MAGE/blob/7e8b52666659529c569487d5278673c3aa04fc61/deployment/utils.py#L280C15-L280C15

    label2decisions = {
        0: "machine-generated",
        1: "human-written",
    }

    # is_machine = - logits[0][0].item()
    is_machine = - logits[0].item()         # we are iterating over one single prediction at the time (so we drop the second indexing [0])
    if is_machine < th:
        decision = 0    # machine generated
    else:
        decision = 1    # human written

    decision_mapper = {
        0: 1,
        1: 0
    }

    return is_machine, decision_mapper[decision]


def main(args):
    device = args.device
    data = get_data(args.datapath)

    if args.split_path is not None:
        test_split = json.load(open(args.split_path))["te"]
        data = pd.DataFrame(data)
        data = data[data["doc-id"].isin(test_split)]
        data = data.to_dict(orient="records")
        output_dir = os.path.join("evaluation_code", "evaluations", args.datapath.split("/")[-1].replace(".zip", ""), "mage_detector", args.target)
    else:
        output_dir = os.path.join("evaluation_code", "evaluations", *args.datapath.split("/")[2:-1], "mage_detector", args.target)

    tokenizer = AutoTokenizer.from_pretrained("yaful/MAGE")
    clf = AutoModelForSequenceClassification.from_pretrained("yaful/MAGE").to(device)

    print(f"- Evaluation mage INFO " + "-" * 25)
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

    preds = []
    for batch_text in tqdm(batched(texts, args.batchsize), total=len(texts) // args.batchsize):
        model_inputs = tokenizer(batch_text, max_length=args.max_length, truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = clf(**model_inputs).logits
        preds.extend(logits.cpu().numpy())
    

    soft_conf_mgt, hard_preds = zip(*[mage_detect(p) for p in preds])
    metrics = classification_report(y_true=labels, y_pred=hard_preds, output_dict=True)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "clf_metrics.json"), "w") as jf:
        json.dump(metrics, jf)
    
    df_preds = pd.DataFrame(data={"doc-id": all_ids, "y_pred": hard_preds, "y_true": labels, "conf_mgt": soft_conf_mgt})
    df_preds.to_csv(os.path.join(output_dir, "clf_preds.csv"), index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, default="generation_code/generations/andrea-dpo-iter1-filtered/2025-01-28-18-49/xsum-testset-250128_223602.json")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target", type=str, default="llama")
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--split_path", type=str, default=None)
    args = parser.parse_args()
    main(args)