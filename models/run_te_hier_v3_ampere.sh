#!/bin/bash
#SBATCH -J te_hier_v3
#SBATCH -A DURBIN-SL2-GPU
#SBATCH -p ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load cuda/11.4

# Load Python only if it exists in this environment.
# If this fails, replace with the python module that is visible under rhel8/default-amp.
module avail 2>&1 | grep -q '^python/3\.11\.0-icl' && module load python/3.11.0-icl || true

export PYTHONNOUSERSITE=1

# Hard fail if venv missing
if [ ! -x "${HOME}/venvs/te_hier_v3/bin/python" ]; then
  echo "ERROR: venv python not found at ${HOME}/venvs/te_hier_v3/bin/python"
  echo "       Rebuild venv under the same module Python you use for jobs."
  exit 1
fi

echo "===== MODULES ====="
module list || true

echo "===== SYSTEM PYTHON (from modules) ====="
echo "which python: $(which python || true)"
python --version || true

echo "===== VENV PYTHON ====="
"${HOME}/venvs/te_hier_v3/bin/python" -c "import sys; print('sys.executable:', sys.executable); print('sys.version:', sys.version)"

echo "===== GPU / DRIVER ====="
nvidia-smi || true
nvidia-smi -L || true

echo "===== TORCH CHECK ====="
~/venvs/te_hier_v3/bin/python - <<'PY'
import os, torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "===== RUN TRAINING ====="
"${HOME}/venvs/te_hier_v3/bin/python" train_hierarchical_v3.py -c config_hierarchical_v3.yaml
