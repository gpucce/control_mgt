import argparse
import json
import gzip
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import pandas as pd


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


def __f_importances(coef, names, title, top, feature_filter = None):
    # Sort importances in descending order
    print(feature_filter)
    imp, names, index = zip(*sorted(zip(coef, names, range(len(coef))), reverse=True))
    if not feature_filter: 
        return index[:top], [str(feat) for feat in names[:top]]
    else:
        top_ten_index = []
        top_ten_names = []
        for i, name in zip(index, names):
            if str(name) in feature_filter:
                continue
            else:
                top_ten_index.append(i)
                top_ten_names.append(str(name))
            if len(top_ten_index) == 10:
                break
        return top_ten_index, top_ten_names


def train_model_and_get_top_feature(path, filter='all', older_features=None):
    X_train, y_train, feature_labels = load_data(f"{path}/train_data.json.gz", feat_filter_path=filter)
    X_val, y_val, feature_labels = load_data(f"{path}/val_data.json.gz", feat_filter_path=filter)
    # X_test, y_test, feature_labels = load_data(f"{path}/test_data.json.gz", feat_filter_path=filter)

    model = make_pipeline(MinMaxScaler(), LinearSVC(random_state=42, max_iter=1000, dual=False))

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print("Validation Set Performance:")
    print(classification_report(y_val, y_val_pred, digits=4))
    _, top_10_index = __f_importances(abs(model.named_steps['linearsvc'].coef_[0]), feature_labels, "Test",
                                   top=10, feature_filter=older_features)

    return model, top_10_index, feature_labels

def __get_true_id(ex):
    return ex.split(".")[0]


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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_filter", "-f", action="store_true", default=False,
                        help="Use only filters that can be verbalized")

    args = parser.parse_args()

    splits_path = "data/data_2024_11_08/splits/llama-3.1-8b-instruct-hf_xsum_informed.split.100000.json"
    with open(splits_path, "r") as input_splits:
        splits = json.loads(input_splits.read())

    training_splits = splits['tr']
    del splits
    
    old_profiling_data_path = "profiling_results/iter/1/"

    _, old_top_10_index, old_feature_labels = train_model_and_get_top_feature(old_profiling_data_path,
                                                                          filter=None if not args.feature_filter
                                                                          else "profiling_results/TO_REMOVE.txt")
    print(f"old top_10 features: {old_top_10_index}" )
    
    new_profiling_data_path = "profiling_results/iter/2dpo/"
    
    _, new_top_10_index, new_feature_labels = train_model_and_get_top_feature(new_profiling_data_path,
                                                                          filter=None if not args.feature_filter
                                                                          else "profiling_results/TO_REMOVE.txt",
                                                                          older_features=old_top_10_index)
    print(f"New top_10 features: {new_top_10_index}" )
        

    originals = pd.read_csv(f"data/profiling_data/xsum_original.zip", compression="zip", sep="\t")
    synth = pd.read_csv(f"data/profiling_data/generations_8b_2_iter_dpo.zip", compression="zip", sep="\t")
    print(originals)
    print(synth)

    df = get_top_examples_second_iter(originals, synth, old_top_10_index, new_top_10_index, 0.1, 1000, training_splits)


    print(df)
    df.to_csv("dpo_dataset/data-iter-2/max_difference_top_10_feature_dataset_no_repetition_tr_filtered_wrt_old.csv", index=False)

