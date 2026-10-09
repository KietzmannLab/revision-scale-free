cat > convert_mni305.sh <<'EOF'
#!/bin/bash
#SBATCH --job-name=mni305
#SBATCH --partition=klab-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

export BONNER_CACHING_HOME=/share/klab/labstudents/elherold/scale-free-data/cache/bonner-caching
export BONNER_CACHING_MODE=readonly

cd /share/klab/labstudents/rafshoon/revision-scale-free/manuscript/scripts-laion
uv run python convert_mni305.py
EOF