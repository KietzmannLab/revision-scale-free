#!/bin/bash
#SBATCH -J laion-loader-test
#SBATCH -t 0-00:30:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -o logs/%x_%j.out

cd /share/klab/labstudents/rafshoon/revision-scale-free
source .venv/bin/activate

echo "Running on $(hostname)"
python manuscript/scripts-laion/loader_test.py
echo "Job completed!"
