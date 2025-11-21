#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --job-name=train-probe-single-gpu
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml

set -euo pipefail


export LOCAL_PROBES_DIR=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/probes
export TRANSFORMERS_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache
export HF_HUB_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache

# private keys
export HF_TOKEN="$(cat ~/keys/.hf_token)"


cd /workspace/hallucination_probes/
git pull

uv pip install transformers==4.56.0 --system


export CUDA_VISIBLE_DEVICES=0
python probe/train.py --config=configs/train_config_slim_llama.yaml

# srun --account=large-sc-2 -p normal  --time=01:00:00 --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml --pty bash