# CSD3 deployment — Round-2.5 winner (T3 + weak1 invariance stack)

The existing CSD3 layout (per `models/slurm_submit_hybrid_v5.sh` and
`models/config_hybrid_v5.yaml`) keeps **everything under `~/TEs/`** with the
following pieces already in place:

```
~/TEs/models/                   # all training scripts + slurm scripts run from here
~/TEs/all_vgp_tes.fa            # FASTA (already there from earlier runs)
~/TEs/20260120_features_sf      # label file (already there)
~/.venv/                        # python venv (already set up)
```

The local `data_analysis/` tree is **not** mirrored on CSD3. So the round-2
code is shipped as a self-contained sub-directory `~/TEs/models/round2/`, and
the SLURM script tells it where the data lives via env vars.

---

## 1. What to upload (manual file copy)

Copy these files individually (e.g. via your remote-file GUI / VS Code Remote
explorer / Finder over SSHFS — whatever you normally use). On CSD3 first:

```
mkdir -p ~/TEs/models/round2/_common
mkdir -p ~/TEs/models/logs
```

Then place the following files **exactly** at the destination paths shown.

### A. SLURM submit script (1 file)

| Local file | CSD3 destination |
|---|---|
| `models/slurm_submit_round2_t3_weak1.sh` | `~/TEs/models/slurm_submit_round2_t3_weak1.sh` |

### B. Round-2 driver + notebook (2 files)

| Local file | CSD3 destination |
|---|---|
| `data_analysis/vgp_model_split_fix/v4/round2/run_smoke.py` | `~/TEs/models/round2/run_smoke.py` |
| `data_analysis/vgp_model_split_fix/v4/cluster session/vgp_hybrid_v4_gpu.ipynb` | `~/TEs/models/round2/vgp_hybrid_v4_gpu.ipynb` |

### C. `_common/` package (6 files — copy the whole folder, but skip `__pycache__/`)

| Local file | CSD3 destination |
|---|---|
| `data_analysis/vgp_model_split_fix/v4/round2/_common/__init__.py` | `~/TEs/models/round2/_common/__init__.py` |
| `data_analysis/vgp_model_split_fix/v4/round2/_common/data_prep.py` | `~/TEs/models/round2/_common/data_prep.py` |
| `data_analysis/vgp_model_split_fix/v4/round2/_common/dann.py` | `~/TEs/models/round2/_common/dann.py` |
| `data_analysis/vgp_model_split_fix/v4/round2/_common/group_dro.py` | `~/TEs/models/round2/_common/group_dro.py` |
| `data_analysis/vgp_model_split_fix/v4/round2/_common/models.py` | `~/TEs/models/round2/_common/models.py` |
| `data_analysis/vgp_model_split_fix/v4/round2/_common/phylo_sampler.py` | `~/TEs/models/round2/_common/phylo_sampler.py` |

> **Skip:** `_common/__pycache__/`, `*.json`, `*.log`, `_run_round25.sh`,
> `RESULTS_*.md`, `README.md`, `CSD3_DEPLOY.md` — none are needed on CSD3.

### D. Data (skip if already there)

| Local file | CSD3 destination |
|---|---|
| `data/vgp/all_vgp_tes.fa` | `~/TEs/all_vgp_tes.fa` |
| `data/vgp/20260120_features_sf` | `~/TEs/20260120_features_sf` |

After copying, the layout on CSD3 should be:

```
~/TEs/models/slurm_submit_round2_t3_weak1.sh
~/TEs/models/logs/                              (empty, slurm writes here)
~/TEs/models/round2/run_smoke.py
~/TEs/models/round2/vgp_hybrid_v4_gpu.ipynb
~/TEs/models/round2/_common/__init__.py
~/TEs/models/round2/_common/data_prep.py
~/TEs/models/round2/_common/dann.py
~/TEs/models/round2/_common/group_dro.py
~/TEs/models/round2/_common/models.py
~/TEs/models/round2/_common/phylo_sampler.py
~/TEs/all_vgp_tes.fa
~/TEs/20260120_features_sf
```

---

## 2. How to operate

SSH in, pre-flight check, submit, monitor:

```bash
ssh USER@HOST
cd ~/TEs/models

# Pre-flight: confirm files are where the slurm script expects.
ls -lh slurm_submit_round2_t3_weak1.sh \
       round2/run_smoke.py \
       round2/_common/data_prep.py \
       round2/vgp_hybrid_v4_gpu.ipynb \
       ~/TEs/all_vgp_tes.fa \
       ~/TEs/20260120_features_sf
ls -lh ~/.venv/bin/python

# Submit.
sbatch slurm_submit_round2_t3_weak1.sh
# -> Submitted batch job <JOBID>

# Monitor queue.
squeue -u $USER

# Live tail (filename uses Slurm's %x_%j = jobname_jobid).
tail -f logs/te_round2_t3_weak1_<JOBID>.out
tail -f logs/te_round2_t3_weak1_<JOBID>.err

# Cancel if needed.
# scancel <JOBID>
```

The slurm script does its own pre-flight (checks venv + each data/notebook
file exists, prints torch/CUDA versions and the 3 TE_* paths) before
launching, so any path mistake fails fast in the first ~30 s of the job.

---

## 3. What it runs

Inside `round2/`:

```
python -u run_smoke.py \
    --track T3 \
    --epochs 10 \
    --subset-size 0 \
    --dann-lambda 0.05 \
    --dann-warmup 3 \
    --gdro-eta 0.005 \
    --tag _full_csd3
```

Hyperparameters are exactly the Round-2.5 winner from
`data_analysis/vgp_model_split_fix/v4/round2/RESULTS_round2_5.md`, scaled up
from smoke (5000 seqs / 6 ep) to full data / 10 ep. `--subset-size 0` is the
"use full dataset" sentinel that was added to `run_smoke.py` in this round.

`data_prep.py` reads the data + notebook locations from these env vars
(exported by the slurm script):

- `TE_NOTEBOOK_PATH` → `~/TEs/models/round2/vgp_hybrid_v4_gpu.ipynb`
- `TE_FASTA_PATH`    → `~/TEs/all_vgp_tes.fa`
- `TE_LABEL_PATH`    → `~/TEs/20260120_features_sf`

If they are unset (e.g. local MPS run), it falls back to the workspace-relative
defaults, so the local smoke pipeline keeps working unchanged.

---

## 4. Output artefacts

After the job completes, fetch from `~/TEs/models/round2/`:

- `smoke_T3_stack_full_csd3.json`  ← test metrics + per-clade hAT recall
- `smoke_T3_stack_full_csd3.log`   ← training log

Pull them back with:

```bash
rsync -av "$CSD3":TEs/models/round2/'smoke_T3_*_full_csd3.*' \
    data_analysis/vgp_model_split_fix/v4/round2/csd3/
```

Compare against the smoke baseline in `RESULTS_round2_5.md`:

| Metric | Smoke (5000, 6 ep) | Target (full, 10 ep) |
|---|---|---|
| cls F1 | 0.577 | ≥ 0.60 |
| SF F1 | 0.298 | ≥ 0.32 |
| hAT F1 | 0.630 | ≥ 0.65 |
| clade-`a` hAT recall | 0.571 | ≥ 0.55 (no regression) |

If clade-`a` hAT recall drops below ~0.4, treat it as a regression and
re-check whether the invariance schedule needs tuning before any further
scaling.
