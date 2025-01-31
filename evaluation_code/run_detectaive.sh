export CUDA_VISIBLE_DEVICES=6

target=llama
datapath=generation_code/generations/adversarial-dpo-iter1-filtered-zscore/2025-01-30-15-28/xsum-testset-250130_163152.json
model=meta-llama/Llama-3.1-8B-Instruct

python evaluation_code/fastdetect_detectgpt.py \
        --datapath $datapath \
        --max_length 256 \
        --batchsize 128 \
        --target $target \
        --base_model_name $model \
        --n_samples 100 
