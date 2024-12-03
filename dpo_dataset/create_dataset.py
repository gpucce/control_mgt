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


def load_data(filename, feat_filter='all'):
    with gzip.open(filename, "rt", encoding="utf-8") as f:
        data = json.load(f)
    split = filename.split("/")[-1].split("_")[0]
    feature_labels = data["features"]

    if feat_filter == "all":
        features_to_remove = []
    elif feat_filter == "filter":
        with open("TO_REMOVE.txt") as f:
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
    X_train, y_train, feature_labels = load_data(f"{path}/train_data.json.gz", feat_filter=filter)
    X_val, y_val, feature_labels = load_data(f"{path}/val_data.json.gz", feat_filter=filter)
    # X_test, y_test, feature_labels = load_data(f"{path}/test_data.json.gz", feat_filter=filter)

    model = make_pipeline(MinMaxScaler(), LinearSVC(random_state=42, max_iter=1000, dual=False))

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print("Validation Set Performance:")
    print(classification_report(y_val, y_val_pred, digits=4))
    _, top_10_index = __f_importances(abs(model.named_steps['linearsvc'].coef_[0]), feature_labels, "Test",
                                   top=10)

    return model, top_10_index, feature_labels


def get_examples(originals, synth, feature, max_number_examples):
    max_abs_difference = 0
    max_abs_difference_index = 0
    raw_difference = 0
    for index, row in originals.iterrows():
        if abs(originals.iloc[index][feature] - synth.iloc[index][feature]) > max_abs_difference:
            max_abs_difference = abs(originals.iloc[index][feature] - synth.iloc[index][feature])
            max_abs_difference_index = originals.iloc[index]['identifier']
            raw_difference = originals.iloc[index][feature] - synth.iloc[index][feature]
    print(f"Feature {feature}: {max_abs_difference}")
    print(raw_difference)
    return max_abs_difference_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("profiling_data_path", help="Path to the profiling data folder.")
    parser.add_argument("--feature_filter", "-f", action="store_true", default=False,
                        help="Use only filters that can be verbalized")

    args = parser.parse_args()


    model, top_10_index, feature_labels = train_model_and_get_top_feature(args.profiling_data_path,
                                                                          filter="all" if not args.feature_filter
                                                                          else "filter")
    originals = pd.read_csv(f"data/profiling_data/xsum_original.zip", compression="zip", sep="\t")
    synth = pd.read_csv(f"data/profiling_data/generations_8b_1_iter.zip", compression="zip", sep="\t")
    for feature in top_10_index:
        print(f"{feature} index: {get_examples(originals, synth, feature, 10)}")

    print(top_10_index)
