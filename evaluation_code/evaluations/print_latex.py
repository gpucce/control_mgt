import os
import json

from pprint import pprint as pp

# TODO it is just printing atm

DETECTORS_ORDER = ["mage", "radar", "detectaive"] #, "binoculars", "detect-gpt"]


def main(args):
    print(f"- basedir: {args.datadir}")

    for detector in DETECTORS_ORDER:
        res_fn = os.path.join(args.datadir, f"{detector}_detector", args.target, "clf_metrics.json")
        metrics = json.load(open(res_fn))[args.label]
        print(detector.upper() + "-" * 25)
        target_metric = metrics[args.metric]
        print(f"{target_metric:.2f}")
        print("-" * 45)
        
    pass

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--datadir", type=str, default="evaluation_code/evaluations/andrea-dpo-iter1-filtered/2025-01-28-18-49")
    parser.add_argument("--target", type=str, default="llama")
    parser.add_argument("--label", type=str, default="macro avg")
    parser.add_argument("--metric", type=str, default="f1-score")
    args = parser.parse_args()
    main(args)