#!/bin/bash
#SBATCH --job-name=laion_loader
#SBATCH --output=laion_%j.out
#SBATCH --error=laion_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -e
echo "Running on $(hostname)"

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python -u manuscript/scripts-laion/laion.py

echo "Job completed!"
