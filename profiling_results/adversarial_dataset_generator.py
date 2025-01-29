import os
import re
import json
import random
import joblib

import pandas as pd
import numpy as np

from pprint import pprint as pp
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler 
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

RANDOM_SEED = 42

pd.options.mode.chained_assignment = None


def print_info_creation(args):
    print("Adversarial Dataset Creation INFO " + "-" * 25)
    print(f"- datset name: {args.dataset_name}")
    print(f"- HWT profile data path: {args.profile_path_hwt}")
    print(f"- MGT profile data path: {args.profile_path_mgt}")
    if args.model_path is not None:
        print(f"- loading trained SVM from {args.model_path}")
    print(f"- MGT text data path: {args.mgt_text}")
    print(f"- HWT text data path: {args.hwt_text}")
    print(f"- MGT generation method: {args.mgt_method}")
    print(f"- remove non-verbalized features: {args.filter}")

    print(f"- adversarial dataset (ids) outdir: profiling_results/adversarial_dataset/{args.dataset_name}/selected_ids.csv")

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
    data_mgt = data_mgt[data_mgt.identifier.isin(idx)] #.drop(columns="identifier")
    data_mgt["label"] = 1

    data_hwt = data_hwt[data_hwt.identifier.isin(idx)] #.drop(columns="identifier")
    data_hwt["label"] = 0

    X_feats = pd.concat((data_mgt, data_hwt), axis=0, ignore_index=True)
    X_feats = shuffle(X_feats)

    labels = X_feats["label"].values
    ids = X_feats["identifier"]
    X_feats = X_feats.drop(columns=["identifier", "label"]).values

    return X_feats, labels, ids
    

def get_top_examples(originals, synths, feature, max_number_examples):
    raw_diff = originals[feature] - synths[feature]
    abs_diff = abs(raw_diff)
    ids = originals["identifier"]

    differences_df = pd.DataFrame({"identifier": ids, "abs_difference": abs_diff, "raw_difference": raw_diff}, columns=['identifier', 'abs_difference', 'raw_difference'])
    
    top_differences = differences_df.sort_values(by='abs_difference', ascending=False).head(max_number_examples)
    top_differences['reason'] = feature

    return top_differences


def get_clf(model_path=None):
    if model_path is not None:
        print(f"- loading trained model from: {model_path}")
        model = joblib.load(model_path)
    else:
        model = make_pipeline(MinMaxScaler(), LinearSVC(random_state=42, max_iter=1000, dual=False))
    return model


def __f_importances(coef, names, top):
    # Sort importances in descending order
    imp, names, index = zip(*sorted(zip(coef, names, range(len(coef))), reverse=True))

    return index[:top], [str(feat) for feat in names[:top]]

# TODO integrate
def get_top_examples_second_iter(original, synth, old_top_features, new_top_features, epsilon, num_rows, training_splits):
    
    # Calculate absolute differences
    diff_old = np.abs(original[old_top_features] - synth[old_top_features])
    diff_new = np.abs(original[new_top_features] - synth[new_top_features])
    dfs = []
    
    diff_old_normalized = (diff_old - diff_old.min()) / (diff_old.max() - diff_old.min())

    
    # Iterate through each column in `new`
    for column in new_top_features:
        # Filter rows where differences in `old` are within tolerance
        mask = (diff_old_normalized <= epsilon).all(axis=1)  # Ensure all `old` columns meet the condition
        
        filtered_indices = [
            idx for idx in original[mask].index
            if int(original.loc[idx, 'identifier'].split(".")[0]) in training_splits
        ]
        
        # Filter data based on `filtered_indices`
        filtered_data = original.loc[filtered_indices]

        # Calculate differences for the current column in `new`
        filtered_diff = diff_new.loc[filtered_indices]

        # Select rows where the difference for the current `new` column is maximized
        top_rows = filtered_diff[column].nlargest(num_rows).index
        df = filtered_data.loc[top_rows]
        df['abs_difference'] = diff_new.loc[top_rows, column]
        df['raw_difference'] = original.loc[top_rows, column] - synth.loc[top_rows, column]
        df['normalized_diff_old_mean'] = diff_old_normalized.loc[top_rows].mean(axis=1)  
        df['reason'] = column
        df['identifier'] = df['identifier'].apply(__get_true_id)

        dfs.append(df)

    print(epsilon)
    final_df = pd.concat(dfs, ignore_index=True)
    print(f"Shuffling the dataset, lenght: {len(final_df)}")
    shuffled_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    final_df = shuffled_df[shuffled_df['abs_difference'] > epsilon]
    print(f"Filtered out all rows with the abs difference is too low, len: {len(final_df)}")
    final_df = final_df.drop_duplicates(subset='identifier', keep='first')
    print(f"Filtered out all row duplicates: {len(final_df)}")
    final_df = final_df[['identifier', 'abs_difference', 'raw_difference', 'normalized_diff_old_mean', 'reason']]

    return final_df


# TODO is_second_iter + old_adversarial_feats
def extract_adversarial_dataset(adversarial_feats, data_hwt, data_mgt, split_idx, is_second_iter=False, old_adversarial_feats=None):
    dfs = []

    data_hwt = data_hwt[data_hwt["identifier"].isin(split_idx)]
    data_mgt = data_mgt[data_mgt["identifier"].isin(split_idx)]

    for i, feat in enumerate(adversarial_feats):
        df = get_top_examples(originals=data_hwt, synths=data_mgt, feature=feat, max_number_examples=1000)
        dfs.append(df)
    
    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.drop_duplicates(subset="identifier", keep="first")

    return final_df
    

def create_dpo_dataset(adversarial_ids, clf_pred, hwt_text, mgt_text, mgt_method):
    # TODO: this is hard-coded to work with llama chat template FIXME
    dataset = []

    for i, row in adversarial_ids.iterrows():
        doc_id = row.identifier
        doc_clf_pred = clf_pred[clf_pred["doc-id"] == doc_id]
        hwt_pred = doc_clf_pred[doc_clf_pred.y_true == 0]["y_pred"].item()
        mgt_pred = doc_clf_pred[doc_clf_pred.y_true == 1]["y_pred"].item()

        hwt_instance = hwt_text[hwt_text["doc-id"] == doc_id]
        mgt_instance = mgt_text[mgt_text["doc-id"] == doc_id]

        system, prompt = __extract_systems_and_prompt(hwt_instance.prompt.item())
        json_line = {'prompt': [], 'chosen': [], 'rejected': []}
        json_line['prompt'].append({"role": "system", "content": system})
        json_line['prompt'].append({"role": "user", "content": prompt})

        if not hwt_pred and mgt_pred:
            # hwt_pred == 0 and mgt_pred == 1 -> classifier has correctly separated the two documents
            chosen = hwt_instance["human"].item()
            rejected = mgt_instance[mgt_method].item()
        elif hwt_pred and mgt_pred:
            # hwt_pred == 1 abd mgt_pred == 1 --> human FP and TP machine
            chosen = mgt_instance[mgt_method].item()
            rejected = hwt_instance["human"].item()
        else:
            chosen = hwt_instance["human"].item()
            rejected = mgt_instance[mgt_method].item()

        json_line["chosen"].append({"role": "assistant", "content": chosen})
        json_line["rejected"].append({"role": "assistant", "content": rejected})
        dataset.append(json_line)
    
    return dataset
    

def main(args):
    print_info_creation(args)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    adversarial_ids_outdir = os.path.join("profiling_results", "adversarial_dataset", args.dataset_name)
    os.makedirs(adversarial_ids_outdir, exist_ok=True)

    split_ids = json.load(open(args.split_file))
    profile_mgt = pd.read_csv(args.profile_path_mgt, sep="\t")
    profile_hwt = pd.read_csv(args.profile_path_hwt, sep="\t")

    # convert ids from string to int    
    profile_mgt = convert_ids(profile_mgt)
    profile_hwt = convert_ids(profile_hwt)

    feature_set = [line.strip() for line in open("profiling_results/all_features_order.txt")]

    if args.filter:
        non_verbalized_features = [line.strip() for line in open("profiling_results/TO_REMOVE.txt")]
        feature_set = [f for f in feature_set if f not in non_verbalized_features]

    column_order = ["identifier"] + feature_set

    profile_mgt = fill_missing_feats(profile_mgt, column_order)
    profile_hwt = fill_missing_feats(profile_hwt, column_order)

    X_tr, Y_tr, tr_ids = get_split(profile_mgt, profile_hwt, idx=split_ids["tr"])
    X_val, Y_val, val_ids = get_split(profile_mgt, profile_hwt, idx=split_ids["val"])
    X_te, Y_te, te_ids = get_split(profile_mgt, profile_hwt, idx=split_ids["te"])

    print(f"({X_tr.shape=},{Y_tr.shape=}) - ({X_te.shape=}, {Y_te.shape=})")
    model = get_clf(args.model_path)

    if args.model_path is None:
        model.fit(X_tr, Y_tr)
    
        print(f"- saving trained SVM in: {adversarial_ids_outdir}")
        joblib.dump(model, os.path.join(adversarial_ids_outdir, "svm_pipeline.joblib"))
    
    y_train_pred = model.predict(X_tr)
    tr_clf_report = classification_report(y_true=Y_tr, y_pred=y_train_pred, digits=2, output_dict=True)
    print(f"tr: {tr_clf_report['macro avg']}")

    y_val_pred = model.predict(X_val)
    val_clf_report = classification_report(y_true=Y_val, y_pred=y_val_pred, digits=2, output_dict=True)
    print(f"val: {val_clf_report['macro avg']}")

    y_test_pred = model.predict(X_te)
    te_clf_report = classification_report(y_true=Y_te, y_pred=y_test_pred, digits=2, output_dict=True)
    print(f"te: {te_clf_report['macro avg']}")
    
    with open(os.path.join(adversarial_ids_outdir, "svm_metrics.json"), "w") as jf:
        json.dump({"train": tr_clf_report, "val": val_clf_report, "te": te_clf_report}, jf)

    df_train_preds = pd.DataFrame(data={"doc-id": tr_ids, "y_true": Y_tr, "y_pred": y_train_pred}).sort_index()

    adver_feats_imp, adversarial_feats = __f_importances(
        coef=abs(model.named_steps["linearsvc"].coef_[0]),
        names=feature_set,
        top=10  
    )

    with open(os.path.join(adversarial_ids_outdir, "selected_feats.json"), "w") as jf:
        feat_dict = {feat_name: feat_imp for feat_name, feat_imp in zip(adversarial_feats, adver_feats_imp)}
        json.dump(feat_dict, jf)

    print(f"- Selected top 10 features:")
    pp(adversarial_feats)

    adversarial_ids = extract_adversarial_dataset(
        adversarial_feats=adversarial_feats,
        data_hwt=profile_hwt,
        data_mgt=profile_mgt,
        split_idx=split_ids["tr"])
    
    adversarial_ids.to_csv(os.path.join(adversarial_ids_outdir, "selected_ids.csv"), index=False)

    # storing args to outdir
    with open(os.path.join(adversarial_ids_outdir, "args.json"), "w") as jf:
        json.dump(args.__dict__, jf, indent=2)

    print("- Building Adversarial DPO dataset")
    hwt_text = pd.read_json(args.hwt_text, lines=True)
    mgt_text = pd.read_json(args.mgt_text, lines=True)

    mgt_text = mgt_text[mgt_text["doc-id"].isin(tr_ids)]
    hwt_text = hwt_text[hwt_text["doc-id"].isin(tr_ids)]

    dpo_dataset = create_dpo_dataset(
        adversarial_ids=adversarial_ids,
        hwt_text=hwt_text,
        mgt_text=mgt_text,
        clf_pred=df_train_preds,
        mgt_method=args.mgt_method
    )
    
    # TODO store as zip file
    adversarial_dpo_dataset = os.path.join(adversarial_ids_outdir, "adversarial_dpo_dataset.json")
    print(f"- storing adversarial dataset in: {adversarial_dpo_dataset}")
    with open(adversarial_dpo_dataset, "w") as jf:
        json.dump(dpo_dataset, jf, ensure_ascii=False)    



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="xsum")
    parser.add_argument("--split_file", type=str, default="data/splits/split.100000.json")
    parser.add_argument("--profile_path_mgt", type=str, default="ilc_profiler/parsed/control-iter1/2025-01-23-12-01/llama/profiling_results/llama_doc.out")
    parser.add_argument("--profile_path_hwt", type=str, default="ilc_profiler/parsed/control-iter1/2025-01-23-12-01/human/profiling_results/human_doc.out")
    parser.add_argument("--mgt_text", type=str)
    parser.add_argument("--mgt_method", type=str, default="llama")
    parser.add_argument("--hwt_text", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    # parser.add_argument("--model_outdir", type=str, default="profiling_results/trained_svm/debug")
    parser.add_argument("--filter", action="store_true", help="remove non-verbalized features")
    parser.add_argument("--second_iter", action="store_true", help="set second-iter sampling strategy")
    args = parser.parse_args()

    main(args)