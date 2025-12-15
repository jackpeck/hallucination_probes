#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH -p normal
#SBATCH --time=12:00:00
#SBATCH --job-name=ap70b-linear
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err
#SBATCH --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml

set -euo pipefail


export LOCAL_PROBES_DIR=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/probes
export TRANSFORMERS_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache
export HF_HUB_CACHE=/capstor/scratch/cscs/tkwiecinski/hallucination_probes/hf_cache

# private keys
export HF_TOKEN="$(cat ~/keys/.hf_token)"
export HF_WRITE_TOKEN="$(cat ~/keys/.hf_token)"
if [[ -f ~/keys/.wandb_key ]]; then
    export WANDB_API_KEY="$(cat ~/keys/.wandb_key)"
fi

# change this to your repo path
cd /workspace/hallucination_probes/
git pull

export CUDA_VISIBLE_DEVICES=0
uv pip install --no-deps  . --system

python probe/train.py --config=configs/apertus_70b_linear_j2.yaml

# srun --account=large-sc-2 -p normal  --time=01:00:00 --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml --pty bash