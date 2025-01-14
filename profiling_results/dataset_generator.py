import pandas as pd
import json
from sklearn.utils import shuffle
from tqdm import tqdm
import numpy as np
import random
from copy import deepcopy
import gzip
import sys


def extract_splits(splits_path, profile_gen_path, profile_og_path):
    
    assert splits_path.endswith(".json") and profile_gen_path.endswith(".tsv") and profile_og_path.endswith(".tsv")

    with open(splits_path) as f:
        splits = json.load(f)

    with open("all_features_order.txt") as f:
        all_features_order = f.read().splitlines()
        all_features_order.insert(0, "identifier") # we add the identifier col since every dataframe has it
    
    df_gen = pd.read_csv(profile_gen_path, sep = "\t")
    # adding all features set to 0 and put them in the same order
    for feat in all_features_order:
        if feat not in df_gen.columns.to_list():
            df_gen[feat] = 0
    df_gen = df_gen[all_features_order]
    
    df_og = pd.read_csv(profile_og_path, sep = "\t")
    for feat in all_features_order:
        if feat not in df_og.columns.to_list():
            df_og[feat] = 0
    df_og = df_og[all_features_order]
    assert df_gen.columns.to_list() == df_og.columns.to_list()
    cols = df_gen.columns.to_list()[1:] # we skip the identifier column

    X_train_og, train_indexes_og = match_indexes(df_og, splits["tr"])
    y_train_og = [0 for _ in X_train_og]
    X_train_gen, train_indexes_gen = match_indexes(df_gen, splits["tr"])
    y_train_gen = [1 for _ in X_train_gen]
    X_train = X_train_og + X_train_gen
    y_train = y_train_og + y_train_gen
    train_indexes = train_indexes_og + train_indexes_gen
    print(len(train_indexes), len(X_train), len(y_train))
    assert len(train_indexes) == len(X_train) and len(X_train) == len(y_train)

    X_val_og, val_indexes_og = match_indexes(df_og, splits["val"])
    y_val_og = [0 for _ in X_val_og]
    X_val_gen, val_indexes_gen = match_indexes(df_gen, splits["val"])
    y_val_gen = [1 for _ in X_val_gen]
    X_val = X_val_og + X_val_gen
    y_val = y_val_og + y_val_gen
    val_indexes = val_indexes_og + val_indexes_gen
    assert len(val_indexes) == len(X_val) and len(X_val) == len(y_val)

    X_test_og, test_indexes_og = match_indexes(df_og, splits["te"])
    y_test_og = [0 for _ in X_test_og]
    X_test_gen, test_indexes_gen = match_indexes(df_gen, splits["te"])
    y_test_gen = [1 for _ in X_test_gen] 
    X_test = X_test_og + X_test_gen
    y_test = y_test_og + y_test_gen
    test_indexes = test_indexes_og + test_indexes_gen
    assert len(test_indexes) == len(X_test) and len(X_test) == len(y_test)

    X_train, y_train, train_indexes = shuffle(X_train, y_train, train_indexes)
    X_val, y_val, val_indexes = shuffle(X_val, y_val, val_indexes)
    X_test, y_test, test_indexes = shuffle(X_test, y_test, test_indexes)

    return X_train, y_train, train_indexes, X_val, y_val, val_indexes, X_test, y_test, test_indexes, cols


def match_indexes(df, indexes):
    
    data = []
    indexes_copy = deepcopy(indexes)
    indexes_to_return = []
    for _, row in tqdm(df.iterrows(), total=len(df)): # l'iter del df è lenta in culo (prob ci sono modi più veloci)
        id = int(row["identifier"].split(".")[0])
        feat_vector = row[1:].values # first one is the identifier
        if id in indexes_copy:
            data.append(feat_vector)
            indexes_copy.remove(id) # una queue brutta indecente, ma velocizza
            indexes_to_return.append(id)

    return data, indexes_to_return   

if __name__ == "__main__":

    # args are: splits, generated_text, original, iter

    splits_path = sys.argv[1]
    profile_gen_path = sys.argv[2]
    profile_og_path = sys.argv[3]
    iter = sys.argv[4]

    random_seed = 42

    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # X_train, y_train, train_indexes, X_val, y_val, val_indexes, X_test, y_test, test_indexes, cols = extract_splits("llama-3.1-8b-instruct-hf_xsum_informed.split.100000.json",
    #                                                                                                         f"output_results/generations_8b_{iter}_.tsv",
    #                                                                                                         "output_results/xsum_original.tsv"
    #                                                                                                         )
    X_train, y_train, train_indexes, X_val, y_val, val_indexes, X_test, y_test, test_indexes, cols = extract_splits(splits_path,
                                                                                                            profile_gen_path,
                                                                                                            profile_og_path
                                                                                                            )
    
    train_data = {
    "X_train": [x.tolist() for x in X_train], 
    "y_train": y_train,
    "train_indexes": train_indexes,
    "features": cols
    }

    val_data = {
        "X_val": [x.tolist() for x in X_val],
        "y_val": y_val,
        "val_indexes": val_indexes,
        "features": cols
    }

    test_data = {
        "X_test": [x.tolist() for x in X_test],
        "y_test": y_test,
        "test_indexes": test_indexes,
        "features": cols
    }

    with gzip.open(f"iter/{iter}/train_data.json.gz", "wt", encoding="utf-8") as f:
        json.dump(train_data, f)
    
    del train_data

    with gzip.open(f"iter/{iter}/val_data.json.gz", "wt", encoding="utf-8") as f:
        json.dump(val_data, f)

    del val_data

    with gzip.open(f"iter/{iter}/test_data.json.gz", "wt", encoding="utf-8") as f:
        json.dump(test_data, f)

    del test_data