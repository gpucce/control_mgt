#!/bin/bash

# Print a message to indicate the start of execution
echo "Starting the execution of Python scripts..."

echo "Dataset creation..."

python dataset_generator.py llama-3.1-8b-instruct-hf_xsum_informed.split.100000.json output_results/generations_8b_1_iter.tsv output_results/xsum_original.tsv 1 
python dataset_generator.py llama-3.1-8b-instruct-hf_xsum_informed.split.100000.json output_results/generations_8b_2_iter.tsv output_results/xsum_original.tsv 2lex 
python dataset_generator.py llama-3.1-8b-instruct-hf_xsum_informed.split.100000.json output_results/generations_8b_2_iter_linguistics.tsv output_results/xsum_original.tsv 2ctg
python dataset_generator.py llama-3.1-8b-instruct-hf_xsum_informed.split.100000.json output_results/generations_8b_2_iter_dpo.tsv output_results/xsum_original.tsv 2dpo

echo "Finished dataset creation."
echo "Starting SVM-DT training and inference for each iter..."

echo "Iter 1:"

python classification_svm.py 1 filter 
python classification_dt.py 1 filter 
python classification_svm.py 1 all 
python classification_dt.py 1 all

echo "Iter 2lex:"

python classification_svm.py 2lex filter 
python classification_dt.py 2lex filter 
python classification_svm.py 2lex all 
python classification_dt.py 2lex all

echo "Iter 2ctg:"

python classification_svm.py 2ctg filter 
python classification_dt.py 2ctg filter 
python classification_svm.py 2ctg all 
python classification_dt.py 2ctg all

echo "Iter 2dpo:"

python classification_svm.py 2dpo filter 
python classification_dt.py 2dpo filter 
python classification_svm.py 2dpo all 
python classification_dt.py 2dpo all

echo "Testing original svm on the new generated texts..."

python classification_svm.py 2lex filter iter/1/svm_pipeline.joblib
python classification_svm.py 2ctg filter iter/1/svm_pipeline.joblib
python classification_svm.py 2dpo filter iter/1/svm_pipeline.joblib