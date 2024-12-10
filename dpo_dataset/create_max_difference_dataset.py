import argparse
import json
import gzip
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.pyplot as plt
import sys


def load_data(filename, feat_filter_path=None):
    with gzip.open(filename, "rt", encoding="utf-8") as f:
        data = json.load(f)
    split = filename.split("/")[-1].split("_")[0]
    feature_labels = data["features"]

    if feat_filter_path is None:
        features_to_remove = []
    else:
        with open(feat_filter_path) as f:
            features_to_remove = f.read().splitlines()

    indexes_to_remove = [feature_labels.index(i) for i in features_to_remove]
    feature_labels = np.delete(feature_labels, indexes_to_remove)
    X = [x for x in data[f"X_{split}"]]  # x
    X = np.delete(X, indexes_to_remove, axis=1)
    y = data[f"y_{split}"]  # y
    return X, y, feature_labels


def __f_importances(coef, names, title, top):
    # Sort importances in descending order
    imp, names, index = zip(*sorted(zip(coef, names, range(len(coef))), reverse=True))

    return index[:top], [str(feat) for feat in names[:top]]


def train_model_and_get_top_feature(path, filter='all'):
    X_train, y_train, feature_labels = load_data(f"{path}/train_data.json.gz", feat_filter_path=filter)
    X_val, y_val, feature_labels = load_data(f"{path}/val_data.json.gz", feat_filter_path=filter)
    # X_test, y_test, feature_labels = load_data(f"{path}/test_data.json.gz", feat_filter_path=filter)

    model = make_pipeline(MinMaxScaler(), LinearSVC(random_state=42, max_iter=1000, dual=False))

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print("Validation Set Performance:")
    print(classification_report(y_val, y_val_pred, digits=4))
    _, top_10_index = __f_importances(abs(model.named_steps['linearsvc'].coef_[0]), feature_labels, "Test",
                                   top=10)

    return model, top_10_index, feature_labels

def __get_true_id(ex):
    return ex.split(".")[0]


def get_top_examples(originals, synth, feature, max_number_examples, training_splits):
    # Create a DataFrame to store the differences and identifiers
    differences = []

    for index, row in originals.iterrows():
        if int(row['identifier'].split(".")[0]) in training_splits:
            difference = abs(originals.iloc[index][feature] - synth.iloc[index][feature])
            raw_difference = originals.iloc[index][feature] - synth.iloc[index][feature]
            identifier = originals.iloc[index]['identifier']
            differences.append((identifier, difference, raw_difference))

    # Convert to a DataFrame for sorting
    differences_df = pd.DataFrame(differences, columns=['identifier', 'abs_difference', 'raw_difference'])

    # Sort by absolute difference in descending order
    top_differences = differences_df.sort_values(by='abs_difference', ascending=False).head(max_number_examples)
    top_differences['identifier'] = top_differences['identifier'].apply(__get_true_id)

    print(f"Top {max_number_examples} differences for feature '{feature}':")
    top_differences['reason'] = feature
    print(top_differences)

    return top_differences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("profiling_data_path", help="Path to the profiling data folder.")
    parser.add_argument("--feature_filter", "-f", action="store_true", default=False,
                        help="Use only filters that can be verbalized")

    args = parser.parse_args()

    splits_path = "data/data_2024_11_08/splits/llama-3.1-8b-instruct-hf_xsum_informed.split.100000.json"
    with open(splits_path, "r") as input_splits:
        splits = json.loads(input_splits.read())
        
    training_splits = splits['tr']
    del splits

    model, top_10_index, feature_labels = train_model_and_get_top_feature(args.profiling_data_path,
                                                                          filter=None if not args.feature_filter
                                                                          else "profiling_results/TO_REMOVE.txt")
    print(top_10_index)

    originals = pd.read_csv(f"data/profiling_data/xsum_original.zip", compression="zip", sep="\t")
    synth = pd.read_csv(f"data/profiling_data/generations_8b_1_iter.zip", compression="zip", sep="\t")
    print(originals)
    print(synth)
    dfs = []

    # Loop through the features and get DataFrames
    for index, feature in enumerate(top_10_index):
        df = get_top_examples(originals, synth, feature, 1000, training_splits)
        dfs.append(df)  # Append the DataFrame to the list

    # Concatenate all DataFrames in the list into a single DataFrame
    final_df = pd.concat(dfs, ignore_index=True)
    shuffled_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_df = shuffled_df.drop_duplicates(subset='identifier', keep='first')

    print(final_df)
    final_df.to_csv("dpo_dataset/data/max_difference_top_10_feature_dataset_no_repetition_tr.csv", index=False)

