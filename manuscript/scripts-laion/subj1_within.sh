#!/bin/bash
#SBATCH --job-name=subj1_within
#SBATCH --output=subj1_within_%j.out
#SBATCH --error=subj1_within_%j.err
#SBATCH --partition=klab-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -e
echo "Running on $(hostname)"

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
python -u manuscript/scripts-laion/subj1_within.py

echo "Job completed!"