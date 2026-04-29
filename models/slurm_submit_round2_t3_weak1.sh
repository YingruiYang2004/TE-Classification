#!/bin/bash
#SBATCH -J te_round2_t3_weak1
#SBATCH -A DURBIN-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/%u/TEs/models/round2/logs/%x_%j.out
#SBATCH --error=/home/%u/TEs/models/round2/logs/%x_%j.err

# Round-2.5 winner: T3 (V4 + CompResEncoder replaces GNN tower) + weak1
# invariance stack (DANN lambda=0.05, GDRO eta=0.005, warmup=3 epochs).
# See data_analysis/vgp_model_split_fix/v4/round2/RESULTS_round2_5.md.
#
# Expected CSD3 layout (mirrors the existing v5 setup; no data_analysis tree
# needed):
#   ~/TEs/models/slurm_submit_round2_t3_weak1.sh   <- this script
#   ~/TEs/models/round2/run_smoke.py               <- code (uploaded fresh)
#   ~/TEs/models/round2/_common/*.py               <- code
#   ~/TEs/models/round2/vgp_hybrid_v4_gpu.ipynb    <- notebook (copied here)
#   ~/TEs/all_vgp_tes.fa                           <- already on CSD3
#   ~/TEs/20260120_features_sf                     <- already on CSD3
#   ~/.venv/bin/python                             <- already on CSD3

set -euo pipefail

SCRIPT_DIR="${HOME}/TEs/models"
ROUND2_DIR="${SCRIPT_DIR}/round2"
cd "$ROUND2_DIR"
mkdir -p logs

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load cuda/11.4
module avail 2>&1 | grep -q '^python/3\.11\.0-icl' && module load python/3.11.0-icl || true

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

# Tell data_prep.py where the data + notebook are on CSD3 (no data_analysis
# tree mirrored here, so the relative-path defaults won't resolve).
export TE_NOTEBOOK_PATH="${ROUND2_DIR}/vgp_hybrid_v4_gpu.ipynb"
export TE_FASTA_PATH="${HOME}/TEs/all_vgp_tes.fa"
export TE_LABEL_PATH="${HOME}/TEs/20260120_features_sf"

VENV_PY="${HOME}/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: venv python not found at $VENV_PY"
  exit 1
fi
for f in "$TE_NOTEBOOK_PATH" "$TE_FASTA_PATH" "$TE_LABEL_PATH"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: required file missing: $f"
    exit 1
  fi
done

echo "===== GPU ====="
nvidia-smi -L

echo "===== TORCH ====="
"$VENV_PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)"

echo "===== PATHS ====="
echo "TE_NOTEBOOK_PATH=$TE_NOTEBOOK_PATH"
echo "TE_FASTA_PATH=$TE_FASTA_PATH"
echo "TE_LABEL_PATH=$TE_LABEL_PATH"

echo "===== RUN T3 weak1 (full data, 10 epochs) ====="
cd "$ROUND2_DIR"
"$VENV_PY" -u run_smoke.py \
    --track T3 \
    --epochs 10 \
    --subset-size 0 \
    --dann-lambda 0.05 \
    --dann-warmup 3 \
    --gdro-eta 0.005 \
    --tag _full_csd3
