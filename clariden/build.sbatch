#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH -p debug
#SBATCH --time=00:40:00
#SBATCH --job-name=build-uv-hallucination
#SBATCH --output=log/build-uv-hallucination-%j.out

set -euo pipefail

cd /iopsstor/scratch/cscs/tkwiecinski/hallucination_probes/clariden
podman build -t uv-hallucination:py-3.10 .

cd /capstor/scratch/cscs/tkwiecinski/hallucination-probes/

rm -f base.sqsh

enroot import -o base.sqsh podman://uv-hallucination:py-3.10
