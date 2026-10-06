#!/bin/bash
#SBATCH --job-name=initial_exploration
#SBATCH --output=initial_exploration_%j.out
#SBATCH --error=initial_exploration_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -e
echo "Running on $(hostname)"

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
python -u manuscript/scripts-laion/initial_exploration.py

echo "Job completed!"
