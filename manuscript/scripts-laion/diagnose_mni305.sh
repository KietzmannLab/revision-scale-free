#!/bin/bash
#SBATCH --job-name=diagnose_mni305
#SBATCH --partition=klab-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

export BONNER_CACHING_HOME=/share/klab/labstudents/elherold/scale-free-data/cache/bonner-caching
export BONNER_CACHING_MODE=readonly

cd /share/klab/labstudents/rafshoon/revision-scale-free/manuscript/scripts-laion
uv run python diagnose_mni305.py
