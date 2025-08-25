Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---- knobs ----
$GPU = 0
$LOGBASE = "runs\activity_sanity"
$OBS_MS = 6000
$BS = 32
$EPOCHS = 70
$PATIENCE = 3

# ---- paths/logs ----
$RepoRoot = (Get-Location).Path
$DateTag = Get-Date -Format "yyyyMMdd_HHmmss"
$JobLogDir = Join-Path $RepoRoot ("joblogs\" + $DateTag)
New-Item -ItemType Directory -Force -Path $JobLogDir | Out-Null
New-Item -ItemType Directory -Force -Path $LOGBASE | Out-Null

function Run-JobAtArgs {
  param(
    [string]$Tag,
    [string]$Dir,
    [string]$Exe,
    [string[]]$ArgList 
  )
  $log = Join-Path $JobLogDir ($Tag + ".log")
  $tb  = Join-Path $LOGBASE $Tag
  New-Item -ItemType Directory -Force -Path $tb | Out-Null

  Write-Host "--------------------------------------------------------------------------------"
  Write-Host "[RUN @ $Dir] $Exe $($ArgList -join ' ')"
  Write-Host " Logs: $log"
  Write-Host " TB  : $tb"
  Write-Host "--------------------------------------------------------------------------------"

  Push-Location $Dir
  try {
    & $Exe @ArgList 2>&1 | Tee-Object -FilePath $log
  } finally {
    Pop-Location
  }
}

# ---- locate tPatchGNN runner ----
$TP_DIR = $null; $TP_SCRIPT = $null
if (Test-Path "model\run_models.py") {
  $TP_DIR = "model"; $TP_SCRIPT = "run_models.py"
} elseif (Test-Path "tPatchGNN\run_models.py") {
  $TP_DIR = "tPatchGNN"; $TP_SCRIPT = "run_models.py"
} else {
  Write-Error "Couldn't find run_models.py (checked model\ and tPatchGNN\)."
  exit 1
}

Write-Host "==> Activity sanity runs (GPU=$GPU)"
Write-Host "==> Job logs: $JobLogDir"

# ------------------- FLD -------------------
$tb_fld = Join-Path $LOGBASE "FLD_activity"
Run-JobAtArgs "FLD_activity" "FLD" "python" @(
  "train_FLD.py",
  "--dataset","activity","-ot",$OBS_MS.ToString(),"-bs",$BS.ToString(),
  "-e",$EPOCHS.ToString(),"-es",$PATIENCE.ToString(),
  "--gpu",$GPU.ToString(),"--tbon","--logdir",$tb_fld
)

# ------------------- IC-FLD (no R0CF) -------------------
$tb_icfld = Join-Path $LOGBASE "ICFLD_activity"
Run-JobAtArgs "ICFLD_activity" "FLD_ICC" "python" @(
  "train_FLD_ICC.py",
  "--dataset","activity","-ot",$OBS_MS.ToString(),"-bs",$BS.ToString(),
  "--epochs",$EPOCHS.ToString(),"--early-stop",$PATIENCE.ToString(),
  "-fn","L","-ed","64","-ld","64","-nh","2","--depth","2",
  "--gpu",$GPU.ToString(),"--tbon","--logdir",$tb_icfld
)

# ------------------- GraFITi -------------------
$tb_graf = Join-Path $LOGBASE "GraFITi_activity"
Run-JobAtArgs "GraFITi_activity" "grafiti" "python" @(
  "train_grafiti.py",
  "--dataset","activity","-ot",$OBS_MS.ToString(),"-bs",$BS.ToString(),
  "--epochs",$EPOCHS.ToString(),"--early-stop",$PATIENCE.ToString(),
  "--gpu",$GPU.ToString(),"--encoder","gratif",
  "--tbon","--logdir",$tb_graf
)

# ------------------- mTAN -------------------
$tb_mtan = Join-Path $LOGBASE "mTAN_activity"
Run-JobAtArgs "mTAN_activity" "mtan" "python" @(
  "train_mtan.py",
  "--dataset","activity","-ot",$OBS_MS.ToString(),"-bs",$BS.ToString(),
  "--niters",$EPOCHS.ToString(),"--early-stop",$PATIENCE.ToString(),
  "--gpu",$GPU.ToString(),
  "--latent-dim","16","--rec-hidden","32","--gen-hidden","50",
  "--embed-time","128","--k-iwae","5","--lr","1e-3",
  "--tbon","--logdir",$tb_mtan
)

# ------------------- t-PatchGNN -------------------
$tb_tpg = Join-Path $LOGBASE "tPatchGNN_activity"
Run-JobAtArgs "tPatchGNN_activity" $TP_DIR "python" @(
  $TP_SCRIPT,
  "--dataset","activity","--history",$OBS_MS.ToString(),
  "--patch_size","1000","--stride","1000",
  "--batch_size",$BS.ToString(),"--epoch",$EPOCHS.ToString(),
  "--patience",$PATIENCE.ToString(),
  "--gpu",$GPU.ToString(),"--tbon","--logdir",$tb_tpg
)

Write-Host "==> All jobs finished. Logs in $JobLogDir"
Write-Host "Open TensorBoard:  tensorboard --logdir $LOGBASE"
