Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$GPU = 0
$LOGDIR = "runs"

# Absolute repo root + absolute joblogs dir
$RepoRoot = (Get-Location).Path
$DateTag = Get-Date -Format "yyyyMMdd_HHmmss"
$JobLogDir = Join-Path $RepoRoot ("joblogs\" + $DateTag)
New-Item -ItemType Directory -Force -Path $JobLogDir | Out-Null

function Run-JobAt {
  param([string]$Tag, [string]$Dir, [string]$CmdLine)
  $log = Join-Path $JobLogDir ($Tag + ".log")   # absolute path now
  Write-Host "--------------------------------------------------------------------------------"
  Write-Host "[RUN @ $Dir] $CmdLine"
  Write-Host "--------------------------------------------------------------------------------"
  Push-Location $Dir
  try {
    Invoke-Expression $CmdLine 2>&1 | Tee-Object -FilePath $log
  } finally {
    Pop-Location
  }
}

# Resolve tPatchGNN runner folder + script name
$TP_DIR = $null; $TP_SCRIPT = $null
if (Test-Path "model\run_models.py") {
  $TP_DIR = "model"; $TP_SCRIPT = "run_models.py"
} elseif (Test-Path "tPatchGNN\run_models.py") {
  $TP_DIR = "tPatchGNN"; $TP_SCRIPT = "run_models.py"
} else {
  Write-Error "Couldn't find run_models.py (checked model\ and tPatchGNN\)."
  exit 1
}

Write-Host "==> Starting all physionet runs (GPU=$GPU, LOGDIR=$LOGDIR)"
Write-Host "==> Logs: $JobLogDir"

# FLD (run from FLD/)
Run-JobAt "FLD_physionet" "FLD" @"
python train_FLD.py -d physionet -ot 24 -bs 32 -e 100 -es 10 --gpu $GPU --tbon --logdir $LOGDIR
"@

# FLD-ICC (run from FLD_ICC/)
Run-JobAt "FLD_ICC_physionet" "FLD_ICC" @"
python train_FLD_ICC.py -d physionet -ot 24 -bs 32 --epochs 100 --early-stop 10 --gpu $GPU -fn L -ed 64 -ld 64 -nh 2 --tbon --logdir $LOGDIR
"@

# GraFITi (run from grafiti/)
Run-JobAt "GraFITi_physionet" "grafiti" @"
python train_grafiti.py -d physionet -ot 24 -bs 32 --epochs 100 --early-stop 10 --gpu $GPU --encoder gratif --tbon --logdir $LOGDIR
"@

# mTAN (run from mtan/)
Run-JobAt "mTAN_physionet" "mtan" @"
python train_mtan.py -d physionet -ot 24 -bs 32 --niters 100 --early-stop 10 --gpu $GPU --latent-dim 16 --rec-hidden 32 --gen-hidden 50 --embed-time 128 --k-iwae 10 --lr 1e-3 --tbon --logdir $LOGDIR
"@

# t-PatchGNN (run from its own folder)
Run-JobAt "tPatchGNN_physionet" $TP_DIR @"
python $TP_SCRIPT --dataset physionet --history 24 --patch_size 8 --stride 8 --batch_size 32 --epoch 100 --patience 10 --gpu $GPU --tbon --logdir $LOGDIR
"@

Write-Host "==> All jobs finished. Logs in $JobLogDir"
Write-Host "Open TensorBoard:  tensorboard --logdir $LOGDIR"
