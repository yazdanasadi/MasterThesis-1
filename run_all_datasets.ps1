Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -------------------- GLOBAL SETTINGS --------------------
$GPU    = 0
$BS     = 32
$EPOCHS = 20
$ES     = 20
$LR     = 1e-3
$WD     = 0
$OBS    = 24   # observation window (hours)

# Per-dataset IC-FLD sizes (from your best runs)
# ed = embedding dim, ld = latent dim, nh = heads
$HP = @{
  "physionet" = @{ ed=64;  ld=32;  nh=8 }
  "mimic"     = @{ ed=32;  ld=128; nh=8 }
  "ushcn"     = @{ ed=32;  ld=64;  nh=4 }
}

# Datasets to run
$DATASETS = @("physionet","mimic","ushcn")

# Detect tPatchGNN entry
$TP_DIR = $null; $TP_SCRIPT = $null
if (Test-Path "model\run_models.py") {
  $TP_DIR = "model"; $TP_SCRIPT = "run_models.py"
} elseif (Test-Path "tPatchGNN\run_models.py") {
  $TP_DIR = "tPatchGNN"; $TP_SCRIPT = "run_models.py"
} else {
  Write-Error "Couldn't find run_models.py (checked model\ and tPatchGNN\)."
  exit 1
}

# Logs
$RepoRoot = (Get-Location).Path
$DateTag  = Get-Date -Format "yyyyMMdd_HHmmss"
$JobLogDir = Join-Path $RepoRoot ("joblogs\" + $DateTag)
New-Item -ItemType Directory -Force -Path $JobLogDir | Out-Null

function Run-JobAt {
  param([string]$Tag, [string]$Dir, [string]$CmdLine)

  $logOut = Join-Path $JobLogDir ($Tag + ".log")
  $logErr = Join-Path $JobLogDir ($Tag + ".err")

  Write-Host "--------------------------------------------------------------------------------"
  Write-Host "[RUN @ $Dir] $CmdLine"
  Write-Host "--------------------------------------------------------------------------------"

  Push-Location $Dir
  try {
    $proc = Start-Process -FilePath "powershell" `
      -ArgumentList @("-NoProfile","-Command",$CmdLine) `
      -NoNewWindow -PassThru `
      -RedirectStandardOutput $logOut `
      -RedirectStandardError  $logErr
    $proc.WaitForExit()
    if ($proc.ExitCode -ne 0) {
      Write-Warning "Run failed (code $($proc.ExitCode)) -> $logOut (stderr: $logErr)"
    }
    if (Test-Path $logOut) { Get-Content $logOut }
    if (Test-Path $logErr) {
      $errTxt = (Get-Content $logErr) -join "`n"
      if ($errTxt.Trim()) { Write-Host "`n[stderr from $Tag]`n$errTxt" -ForegroundColor Yellow }
    }
  } finally { Pop-Location }
}

Write-Host "==> Starting all datasets (GPU=$GPU)"
Write-Host "==> Logs: $JobLogDir"

foreach ($ds in $DATASETS) {
  if (-not $HP.ContainsKey($ds)) { Write-Warning "Skip unknown dataset $ds"; continue }
  $h = $HP[$ds]
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"

  # -------------------- FLD (supports -fn) --------------------
  # $fldDir = "runs/$ds/FLD/$stamp"
  # $cmd = "python train_FLD.py -d $ds -ot $OBS -bs $BS -e $EPOCHS -es $ES --gpu $GPU --tbon --logdir `"$fldDir`" -fn L"
  # Run-JobAt "FLD_$ds" "FLD" $cmd

#   # -------------------- IC-FLD (NO CYCLE; no --resume, no depth flag) --------------------
#   if ($ds -eq "ushcn") {
#     # use your USHCN-specific trainer
#     $nocDir = "runs/$ds/ICFLD_noRCF/$stamp"
#     $cmd = "python train_FLD_ICC_ushcn.py -d $ds -ot $OBS -bs $BS -e $EPOCHS -es $ES " +
#            "-fn L -ed $($h.ed) -ld $($h.ld) -nh $($h.nh) " +  # no -dp, no --resume
#            "--gpu $GPU --tbon --logdir `"$nocDir`""
#     Run-JobAt "ICFLD_noRCF_$ds" "FLD_ICC" $cmd
#   } else {
#     # generic IC-FLD trainer
#     $nocDir = "runs/$ds/ICFLD_noRCF/$stamp"
#     $cmd = "python train_FLD_ICC.py -d $ds -ot $OBS -bs $BS --epochs $EPOCHS --early-stop $ES --gpu $GPU " +
#            "-fn L -ed $($h.ed) -ld $($h.ld) -nh $($h.nh) " +  # no --depth, no --resume
#            "--lr $LR --wd $WD --tbon --logdir `"$nocDir`""
#     Run-JobAt "ICFLD_noRCF_$ds" "FLD_ICC" $cmd
#   }

  # -------------------- GraFITi --------------------
  # $grDir = "runs/$ds/GraFITi/$stamp"
  # $cmd = "python train_grafiti.py -d $ds -ot $OBS -bs $BS --epochs $EPOCHS --early-stop $ES --gpu $GPU --encoder gratif --tbon --logdir `"$grDir`""
  # Run-JobAt "GraFITi_$ds" "grafiti" $cmd

  # -------------------- mTAN --------------------
  # Map IC-FLD sizes => mTAN:
  # latent-dim := ld, rec-hidden := 2*ld, gen-hidden := 2*ld, embed-time := ed
  $mtanDir = "runs/$ds/mTAN/$stamp"
  $recH = [int](2*$h.ld); $genH = $recH
  $cmd = "python train_mtan.py -d $ds -ot $OBS -bs $BS --niters $EPOCHS --early-stop $ES --gpu $GPU " +
         "--latent-dim $($h.ld) --rec-hidden $recH --gen-hidden $genH --embed-time $($h.ed) --k-iwae 10 " +
         "--lr $LR --tbon --logdir `"$mtanDir`""
  Run-JobAt "mTAN_$ds" "mtan" $cmd

  # -------------------- t-PatchGNN --------------------
  # Map IC-FLD sizes => tPatchGNN:
  # hid_dim := ld, te_dim := ed, node_dim := ed, nhead := nh; keep layers defaults
  $tpDir = "runs/$ds/tPatchGNN/$stamp"
  $cmd = "python $TP_SCRIPT --dataset $ds --history $OBS --patch_size 8 --stride 8 --batch_size $BS " +
         "--epoch $EPOCHS --patience $ES --gpu $GPU --tbon --logdir `"$tpDir`" " +
         "--hid_dim $($h.ld) --te_dim $($h.ed) --node_dim $($h.ed) --nhead $($h.nh)"
  Run-JobAt "tPatchGNN_$ds" $TP_DIR $cmd
}

Write-Host "==> All jobs finished. Logs in $JobLogDir"
Write-Host "Open TensorBoard (separate folders):  tensorboard --logdir runs"
