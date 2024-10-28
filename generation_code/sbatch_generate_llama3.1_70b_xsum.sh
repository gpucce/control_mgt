#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=4
#SBATCH --time=0-12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_E03_068
#SBATCH --job-name=generate_llama3.1_70b
#SBATCH --output=slurm_logs/llm_llama3_sender-%j.out


eval "$(conda shell.bash hook)" # init conda
conda activate /leonardo_scratch/large/userexternal/gpuccett/Repos/MGT2025-private/conda_venv_vllm
module load gcc

srun python /leonardo_scratch/large/userexternal/gpuccett/Repos/MGT2025-private/humaneval/it/generation_code/generate_article_eng_xsum.py \
    --model_name llama-3.1-70b-instruct-hf --informed
