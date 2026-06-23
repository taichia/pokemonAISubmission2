# Environment setup

## What this code actually needs

- **Python 3.10+** (the code uses `list[int]` annotations; Kaggle runs 3.10/3.11).
- **PyTorch** — the *only* third-party package. Everything else is the standard library.
- **The `cg/` simulator** — already in the repo as a precompiled native library
  (`cg.dll` on Windows, `libcg.so` on Linux). Loaded via `ctypes`, nothing to install.
  Verified loading + `GameInitialize()` succeeds.

CUDA is **optional**. Every device line in the code is
`torch.device("cuda" if torch.cuda.is_available() else "cpu")`, so it runs on CPU
with no changes. The network is small (2-layer transformer, d_model=128), so CPU is
fine for inference (`main.py`) and usable for training, just slower. A GPU mainly
speeds up `ppo_training.py` / `bc_pretrain.py`.

## Install (Windows)

From the project folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Option A — CPU only (simplest, always works)

```powershell
pip install torch
```

### Option B — GPU / CUDA (faster training)

1. Check you have an NVIDIA GPU and driver: run `nvidia-smi`. If that command isn't
   found, you don't have a usable NVIDIA setup — use Option A.
2. Install the CUDA build of torch (cu124 shown; pick the index matching your driver
   from https://pytorch.org/get-started/locally/):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

You do **not** need to install the CUDA Toolkit separately — the pip wheel bundles the
CUDA runtime. You only need a recent NVIDIA driver.

## Verify it works

```powershell
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import cg.api; print('simulator OK')"
```

If both print without error, the environment is ready.

## Running

- `main.py` (inference) needs trained weights at `out/best.pth`. It raises
  `FileNotFoundError` if missing — produce them with `ppo_training.py` first, or drop
  an existing checkpoint there.
- `python bc_pretrain.py` — behavior-cloning pretrain.
- `python ppo_training.py` — PPO training (uses multiprocessing; GPU recommended).
