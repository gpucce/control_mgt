import pandas as pd


if __name__ == "__main__":
    old_df = pd.read_csv("dpo_dataset/data-iter-1/max_difference_top_10_feature_dataset_no_repetition_tr.csv")
    new_df = pd.read_csv("dpo_dataset/data-iter-2/max_difference_top_10_feature_dataset_no_repetition_tr_filtered_wrt_old.csv")
    
    old_identifiers = old_df['identifier']
    print("Filtering old sentencing from re-appearing, before length: " + str(len(new_df)))
    new_df = new_df[~new_df['identifier'].isin(old_identifiers)]
    print(f"After len: {len(new_df)}")
    new_df.to_csv("dpo_dataset/data-iter-2/max_difference_top_10_feature_dataset_no_repetition_tr_filtered_wrt_old.csv", index=False)
