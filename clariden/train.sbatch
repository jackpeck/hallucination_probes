#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --job-name=train-probe-single-gpu
#SBATCH --output=log/%x-%j.out
#SBATCH --output=log/%x-%j.err
#SBATCH --environment=/iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden/env.toml

set -euo pipefail

cd /workspace/hallucination_probes/
git pull
python probe/train.py --config=configs/train_config_slim_llama.yaml