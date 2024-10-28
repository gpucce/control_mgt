#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=4
#SBATCH --time=0-12:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_E03_068
#SBATCH --job-name=generate_llama3.1_405b
#SBATCH --output=slurm_logs/llm_llama3_sender-%j.out


eval "$(conda shell.bash hook)" # init conda
conda activate /leonardo_scratch/large/userexternal/gpuccett/Repos/MGT2025-private/conda_venv_vllm
module load gcc

export CUDA_VISIBLE_DEVICES="0,1,2,3"

# ===== DO NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING =====
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
# redis_password=$(uuidgen)
# export redis_password


nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
# ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address)

ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" \
  ray start --head --node-ip-address="$ip" --port=$port --block &
sleep 10

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  echo "IP Head: $ip_head"
  this_node_ip=$(srun --nodes=1 --ntasks=1 -w "$node_i" hostname --ip-address)
  echo "Node: $node_i IP Local: $this_node_ip"
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    ray start \
    --address "$ip_head" \
    --node-ip-address ${this_node_ip} \
    --num-cpus 32 \
    --num-gpus 4 \
    --block &

  sleep 5
done

# ===== Call your code below =====

echo "STARTING python command"

python /leonardo_scratch/large/userexternal/gpuccett/Repos/MGT2025-private/humaneval/it/generation_code/generate_article_eng_xsum.py \
    --model_name llama-3.1-405b-instruct-hf --informed
