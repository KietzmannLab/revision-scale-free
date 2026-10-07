#!/bin/bash
#SBATCH --job-name=compare_loaders
#SBATCH --output=compare_loaders_%j.out
#SBATCH --error=compare_loaders_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -e
echo "Running on $(hostname)"

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export BONNER_CACHING_HOME=/share/klab/labstudents/rafshoon/scale-free-data/cache/bonner-caching
python -u manuscript/scripts-laion/compare_loaders.py

echo "Job completed!"
