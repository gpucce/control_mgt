import os
import joblib

import json
import pandas as pd

from sklearn.metrics import classification_report


pd.options.mode.chained_assignment = None


def convert_ids(data):
    data.identifier = data.identifier.str.replace(".conllu", "")
    data.identifier = data.identifier.astype("int")
    return data


def fill_missing_feats(data, all_features):
    for feat in all_features:
        if feat not in data.columns:
            data[feat] = 0
    return data[all_features]


def get_split(data_mgt, data_hwt, idx):
    data_mgt = data_mgt[data_mgt.identifier.isin(idx)] #.drop(columns="identifier")
    data_mgt["label"] = 1

    data_hwt = data_hwt[data_hwt.identifier.isin(idx)] #.drop(columns="identifier")
    data_hwt["label"] = 0

    X_feats = pd.concat((data_mgt, data_hwt), axis=0, ignore_index=True)

    labels = X_feats["label"].values
    ids = X_feats["identifier"]
    X_feats = X_feats.drop(columns=["identifier", "label"]).values

    return X_feats, labels, ids

def get_output_dir(args):
    basedir = os.path.join("evaluation_code", "evaluations")
    model_name, run_name, _ = args.mgt_profile.split("/")[-3:]
    run_name = run_name.replace("-cut256", "")
    dataset_name = "xsum" if "xsum" in args.mgt_profile else "m4abs"
    if "naive" in args.mgt_profile:
        dataset_name += "-naive"
    if "iter1" in args.mgt_profile:
        dataset_name += "-iter-1"
    elif "iter2" in args.mgt_profile:
        dataset_name += "-iter-2"
    else:
        pass
    target = args.mgt_profile.split("/")[-1].replace("_doc.out", "")    
    target = target.replace("testset_", "")
    outdir = os.path.join(basedir, dataset_name, model_name, run_name, "svm_detector", target)
    return outdir, target 

def main(args):
    profile_hwt= pd.read_csv(args.hwt_profile, sep="\t")
    profile_mgt= pd.read_csv(args.mgt_profile, sep="\t")
    testset_ids = json.load(open(args.split_path))["te"]
    feature_set = [line.strip() for line in open("profiling_results/all_features_order.txt")]

    output_dir, target = get_output_dir(args)

    if not args.unfiltered:
        non_verbalized_features = [line.strip() for line in open("profiling_results/TO_REMOVE.txt")]
        feature_set = [f for f in feature_set if f not in non_verbalized_features]
    
    column_order = ["identifier"] + feature_set

    profile_mgt = convert_ids(profile_mgt)
    profile_hwt = convert_ids(profile_hwt)

    profile_mgt = fill_missing_feats(profile_mgt, column_order)
    profile_hwt = fill_missing_feats(profile_hwt, column_order)

    X_te, labels, te_ids = get_split(profile_mgt, profile_hwt, idx=testset_ids)
    
    print(f"- Evaluation support-vector classifier INFO" + "-" * 25)
    print(f"- storing results in: {output_dir}")
    print(f"- target: {target}")
    print(f"- num pairs: {len(X_te) // 2}")
    print("-" * 45)

    model = joblib.load(args.svm_path)
    
    hard_preds = model.predict(X_te)
    soft_preds = model.decision_function(X_te)
    
    metrics = classification_report(y_true=labels, y_pred=hard_preds, output_dict=True)

    print(metrics["macro avg"])

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "clf_metrics.json"), "w") as jf:
        json.dump(metrics, jf)
    
    df_preds = pd.DataFrame(data={"doc-id": te_ids, "y_pred": hard_preds, "y_true": labels, "score": soft_preds})
    df_preds.to_csv(os.path.join(output_dir, "clf_preds.csv"), index=False)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--hwt_profile", type=str, default="ilc_profiler/parsed/xsum/vanilla/human-cut256/human_doc.out")
    parser.add_argument("--mgt_profile", type=str, default="ilc_profiler/parsed/adversarial-dpo-iter1-filtered/2025-01-30-23-48/dpo-llama-1st-iter-cut256/dpo-llama-1st-iter_doc.out")
    parser.add_argument("--svm_path", type=str, default="profiling_results/adversarial_dataset/xsum/dpo-iter1-filtered-cut256/svm_pipeline.joblib")
    parser.add_argument("--unfiltered", action="store_true")
    parser.add_argument("--split_path", type=str, default=None)
    # parser.add_argument("--target", type=str, required=True)
    args = parser.parse_args()
    main(args)