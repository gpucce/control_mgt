# https://github.com/ahans30/Binoculars


def main(args):
    raise NotImplementedError

    output_dir = os.path.join("evaluation_code", "evaluations", *args.datapath.split("/")[2:-1], "binoculars_detector", args.target)
    print(f"- Evaluation binoculars INFO" + "-" * 25)
    print(f"- storing results in: {output_dir}")
    print(f"- target: {args.target}")
    print(f"- num pairs: {len(data)}")
    print("-" * 45)

    ### detecion code starts
    ### detection code ends

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "clf_metrics.json"), "w") as jf:
        json.dump(metrics, jf)
    
    # store models output (scores/logits/probs) too! (e.g., {"p_0": soft_0, "p_1": soft_1})
    df_preds = pd.DataFrame(data={"doc-id": all_ids, "y_pred": hard_preds, "y_true": labels, "p_0": soft_0, "p_1": soft_1})
    df_preds.to_csv(os.path.join(output_dir, "clf_preds.csv"), index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--datapath", type=str, default="generation_code/generations/andrea-dpo-iter1-filtered/2025-01-28-18-49/xsum-testset-250128_223602.json")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--target", type=str, default="llama")
    parser.add_argument("--batchsize", type=int, default=64)
    args = parser.parse_args()
    main(args)