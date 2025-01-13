import argparse
import io
import json
import zipfile
import pandas as pd
import re

from create_max_difference_dataset import train_model_and_get_top_feature


def __extract_feature_for_prediction(identifier, df, feature_labels):
    feature_labels = feature_labels.tolist()
    feature_labels.append("identifier")
    
    # Ensure all required columns exist in the DataFrame
    missing_features = [col for col in feature_labels if col not in df.columns]
    if missing_features:
        print(f"Warning: The following features are missing and will be filled with default values: {missing_features}")
    
    # Add missing columns with default values (e.g., 0)
    for col in missing_features:
        df[col] = 0  # Default value can be adjusted as needed
    
    # Filter DataFrame to only include required columns
    df = df[df.columns.intersection(feature_labels)]
    
    # Select the row matching the identifier
    row = df.loc[df['identifier'] == identifier].copy()
    row = row.drop(['identifier'], axis=1)
    
    # Validate feature dimensions
    if row.shape[1] != len(feature_labels) - 1:  # Minus 1 for 'identifier'
        raise ValueError(f"Feature mismatch: Expected {len(feature_labels) - 1}, but got {row.shape[1]}.")
    
    return row.values



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

        # Print the extracted texts
        """print("system_text:")
        print(system_text)
        print("\nuser_prompt:")
        print(user_prompt)"""
        return system_text, user_prompt
    else:
        print("No match found.")
        raise Exception("No System Prompt found")


def create_dataset(max_difference_df, generations_df_path, pUD_originals, pUD_synth, model, feature_labels):
    identifiers = max_difference_df['identifier'].to_list()
    dataset = {
        "system": [],
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }

    with open("dataset-max-feature-difference-top-10_iter_2.jsonl", "w") as output_file:
        with zipfile.ZipFile(generations_df_path) as zf:
            with io.TextIOWrapper(
                    zf.open("generation_output_llama-3.1-8b-instruct-hf_xsum_temp0.8_informed_cut256.jsonl"),
                    encoding="utf-8") as f:
                for row in f:
                    row = json.loads(row)
                    if int(row['id']) in identifiers:
                        og_X = __extract_feature_for_prediction(row['id'] + ".conllu", pUD_originals, feature_labels)
                        synth_X = __extract_feature_for_prediction(row['id'] + ".conllu", pUD_synth, feature_labels)
                        og_pred = model.predict(og_X)[0]
                        synth_pred = model.predict(synth_X)[0]

                        system, prompt = __extract_systems_and_prompt(row['prompt'])
                        json_line = {'prompt': [], 'chosen': [], 'rejected': []}
                        json_line['prompt'].append({"role": "system", "content": system})
                        json_line['prompt'].append({"role": "user", "content": prompt})

                        if not og_pred and synth_pred:  # og predetto bene synth pure
                            json_line['chosen'].append({"role": "assistant", "content": row['real_article']})
                            json_line['rejected'].append({"role": "assistant", "content": row['generated_text']})
                        elif og_pred and not synth_pred:  # synth predetto come reale, og come falso
                            json_line['rejected'].append({"role": "assistant", "content": row['real_article']})
                            json_line['chosen'].append({"role": "assistant", "content": row['generated_text']})
                        else:  # ogni altro caso metto chosen al reale
                            json_line['chosen'].append({"role": "assistant", "content": row['real_article']})
                            json_line['rejected'].append({"role": "assistant", "content": row['generated_text']})
                        output_file.write(json.dumps(json_line) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("profiling_data_path", help="Path to the profiling data folder.")
    parser.add_argument("--feature_filter", "-f", action="store_true", default=False,
                        help="Use only filters that can be verbalized")

    args = parser.parse_args()

    model, top_10_index, feature_labels = train_model_and_get_top_feature(args.profiling_data_path,
                                                                          filter=None if not args.feature_filter
                                                                          else "profiling_results/TO_REMOVE.txt")
    max_difference_df = pd.read_csv("dpo_dataset/data-iter-2/max_difference_top_10_feature_dataset_no_repetition_tr.csv")

    originals = pd.read_csv(f"data/profiling_data/xsum_original.zip", compression="zip", sep="\t")
    synth = pd.read_csv(f"data/profiling_data/generations_8b_2_iter_dpo.zip", compression="zip", sep="\t")

    create_dataset(max_difference_df,
                   "data/data_2024_11_12/generation_output_llama-3.1-8b-instruct-hf_xsum_temp0.8_informed_cut256.zip",
                   originals, synth, model, feature_labels)
