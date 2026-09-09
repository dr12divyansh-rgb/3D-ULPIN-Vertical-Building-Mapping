# Run full experiment pipeline: eval exp003, train+eval exp004, compare.
# Usage: .venv\Scripts\python.exe -u run_experiments_complete.ps1
# (Actually run this as a PowerShell script)

Set-Location "C:\Users\Takin\OneDrive\Desktop\3d-ulpin-project"
$py = ".venv\Scripts\python.exe"

function Run-Step {
    param([string]$desc, [string[]]$cmd)
    Write-Host "`n=== $desc ===" -ForegroundColor Cyan
    & $cmd[0] $cmd[1..($cmd.Length-1)]
    if ($LASTEXITCODE -ne 0) { Write-Warning "Step failed: $desc" }
}

# Find experiment_003 checkpoint
$exp003 = Get-ChildItem ml\results\experiment_003 -Filter "best_model.pth" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $exp003) { Write-Error "exp003 best_model.pth not found"; exit 1 }
$ckpt003 = $exp003.FullName
Write-Host "Exp003 checkpoint: $ckpt003"

# 1. Evaluate exp003 on STPLS3D test
Run-Step "Evaluate exp003 on STPLS3D test" @($py, "-u", "ml/src/evaluate_stpls3d.py", "--checkpoint", $ckpt003, "--data", "ml/data/stpls3d", "--batch-size", "8", "--no-save-ply")

# 2. Train exp004: mixed
Run-Step "Train exp004: mixed synthetic+STPLS3D" @($py, "-u", "ml/src/training/train.py", "--config", "ml/config/mixed_semantic.yaml", "--results", "ml/results")

# Find experiment_004 checkpoint
$exp004 = Get-ChildItem ml\results\experiment_004 -Filter "best_model.pth" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $exp004) { Write-Warning "exp004 best_model.pth not found; skipping evaluation" }
else {
    $ckpt004 = $exp004.FullName
    Write-Host "Exp004 checkpoint: $ckpt004"

    # 3. Evaluate exp004 on STPLS3D test
    Run-Step "Evaluate exp004 on STPLS3D test" @($py, "-u", "ml/src/evaluate_stpls3d.py", "--checkpoint", $ckpt004, "--data", "ml/data/stpls3d", "--batch-size", "8", "--no-save-ply")

    # 4. Evaluate exp004 on synthetic test
    Run-Step "Evaluate exp004 on synthetic test" @($py, "-u", "ml/src/evaluate.py", "--checkpoint", $ckpt004, "--data", "ml/data/synthetic", "--batch-size", "8")

    # 5. Compare all experiments
    Run-Step "Compare exp001 vs exp003 vs exp004" @($py, "-u", "ml/src/evaluation/compare_stpls3d_experiments.py", "--exp001", "ml/results/experiment_001", "--exp003", "ml/results/experiment_003", "--exp004", "ml/results/experiment_004")
}

Write-Host "`n=== PIPELINE COMPLETE ===" -ForegroundColor Green
