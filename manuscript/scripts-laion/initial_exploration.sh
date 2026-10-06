#!/bin/bash
#SBATCH --job-name=initial_exploration
#SBATCH --output=initial_exploration_%j.out
#SBATCH --error=initial_exploration_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

echo "Running on $(hostname)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scale-free-visual-cortex

cd "$SLURM_SUBMIT_DIR"
python -u initial_exploration.py

echo "Job completed!"
