#!/bin/bash
#SBATCH -J tir_v1
#SBATCH -A DURBIN-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Always run from the directory containing this script (~/TEs/models)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

# Modules
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load cuda/11.4

# Load Python only if present in this module environment (optional; venv python is used)
module avail 2>&1 | grep -q '^python/3\.11\.0-icl' && module load python/3.11.0-icl || true

# Unbuffered output helps progress reporting in SLURM logs
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

VENV_PY="${HOME}/venvs/te_hier_v3/bin/python"
CFG="${SCRIPT_DIR}/config_tir_v1.yaml"
SCRIPT="${SCRIPT_DIR}/train_tir_v1.py"

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: venv python not found at $VENV_PY"
  exit 1
fi

echo "===== MODULES ====="
module list || true

echo "===== GPU ====="
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L

echo "===== PYTHON ====="
"$VENV_PY" -c "import sys; print(sys.executable); print(sys.version)"

echo "===== TORCH ====="
"$VENV_PY" -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('torch cuda', torch.version.cuda); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "===== RUN TIR Binary Classifier V1 ====="
"$VENV_PY" -u "$SCRIPT" -c "$CFG"
