#!/bin/bash
#SBATCH --job-name=cross_correlations_to_template
#SBATCH --partition=klab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

export BONNER_CACHING_HOME=/share/klab/labstudents/elherold/scale-free-data/cache/bonner-caching
export BONNER_CACHING_MODE=readonly

cd /share/klab/labstudents/rafshoon/revision-scale-free/manuscript/scripts-laion
export MNI_METHOD=to_template
uv run python cross_correlations_laion.
