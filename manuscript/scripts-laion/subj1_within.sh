#!/bin/bash
#SBATCH --job-name=general_region_laion
#SBATCH --output=general_region_laion_%j.out
#SBATCH --error=general_region_laion_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -e
echo "Running on $(hostname)"

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python -u manuscript/scripts-laion/subj1_within.py

echo "Job completed!"
