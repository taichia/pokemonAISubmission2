@echo off
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"

echo === dry run start === > dry_run_output.log 2>&1
echo using python: %PY% >> dry_run_output.log 2>&1

echo [step1] sanity: torch + engine + deck >> dry_run_output.log 2>&1
"%PY%" -c "import torch, cg.api; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('deck', len(open('deck.csv').read().split()))" >> dry_run_output.log 2>&1

echo [step2] behavioral cloning (tiny) >> dry_run_output.log 2>&1
"%PY%" bc_pretrain.py --games 8 --epochs 1 --eval-games 4 >> dry_run_output.log 2>&1

echo [step3] one PPO iteration warmstarted from BC >> dry_run_output.log 2>&1
"%PY%" ppo_training.py --iters 2 --games 8 --eval-games 4 --workers 1 --resume out/bc.pth --lr 1e-4 >> dry_run_output.log 2>&1

echo [step4] submission agent loads trained weights >> dry_run_output.log 2>&1
"%PY%" main.py >> dry_run_output.log 2>&1

echo === dry run done === >> dry_run_output.log 2>&1
