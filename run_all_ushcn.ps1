# hpsearch_icfld_ushcn.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -------------------- CONFIG --------------------
$Dataset    = "ushcn"
$GPU        = 0
$Obs        = 24
$BatchSize  = 32
$Epochs     = 70
$EarlyStop  = 20
$Fn         = "L"
$BaseLogDir = "runs"

# Sweep space
$Seeds       = @(0)                     # add more if you want: @(0,1,2)
$LatentDims  = @(32, 64, 128, 256)      # -ld
$Heads       = @(2, 4, 8)               # -nh
$Depths      = @(2, 4)                  # -dp
$EmbPerHead  = @(2, 4, 8)               # total embed dim = nh * emb_per_head

# RCF (cycle removal) params
$CycleLength    = 24
$TimeMaxHours   = 48

# Trainers (must exist in FLD_ICC/)
$TrainDir       = "FLD_ICC"
$NoCycleScript  = "train_FLD_ICC_ushcn.py"
$RCFScript      = "train_FLD_ICC_ushcn_rcf.py"

# -------------------- PREP ----------------------
$RepoRoot  = (Get-Location).Path
$DateTag   = Get-Date -Format "yyyyMMdd_HHmmss"
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

    # Echo logs back to console
    if (Test-Path $logOut) { Get-Content $logOut }
    if (Test-Path $logErr) {
      $errTxt = (Get-Content $logErr) -join "`n"
      if ($errTxt.Trim()) {
        Write-Host "`n[stderr from $Tag]`n$errTxt" -ForegroundColor Yellow
      }
    }
  }
  finally {
    Pop-Location
  }
}


# Check scripts exist
if (-not (Test-Path (Join-Path $TrainDir $NoCycleScript))) {
  Write-Error "Missing $TrainDir\$NoCycleScript"
  exit 1
}
if (-not (Test-Path (Join-Path $TrainDir $RCFScript))) {
  Write-Error "Missing $TrainDir\$RCFScript"
  exit 1
}

Write-Host "==> IC-FLD USHCN sweep starting (GPU=$GPU)"
Write-Host "==> Logs: $JobLogDir"
Write-Host ""

# -------------------- SWEEP: NO CYCLE --------------------
Write-Host "### NO-CYCLE RUNS ###"
foreach ($s in $Seeds) {
  foreach ($ld in $LatentDims) {
    foreach ($nh in $Heads) {
      foreach ($eph in $EmbPerHead) {
        $ed = $nh * $eph
        foreach ($dp in $Depths) {
          $tbSubdir = "$BaseLogDir/ushcn/icfld_nocycle/$DateTag/s${s}_ld${ld}_nh${nh}_eph${eph}_ed${ed}_dp${dp}"
          $tag = "ICFLD_USHCN_NC_s${s}_ld${ld}_nh${nh}_eph${eph}_dp${dp}"
          $cmd = @(
            "python $NoCycleScript",
            "-d $Dataset -ot $Obs -bs $BatchSize -e $Epochs -es $EarlyStop",
            "-fn $Fn -ed $ed -ld $ld -nh $nh -dp $dp",
            "--gpu $GPU --tbon --logdir `"$tbSubdir`""
          ) -join " "
          Run-JobAt $tag $TrainDir $cmd
        }
      }
    }
  }
}

# -------------------- SWEEP: WITH CYCLES (RCF) --------------------
Write-Host ""
Write-Host "### RCF (cycle removal) RUNS ###"
foreach ($s in $Seeds) {
  foreach ($ld in $LatentDims) {
    foreach ($nh in $Heads) {
      foreach ($eph in $EmbPerHead) {
        $ed = $nh * $eph
        foreach ($dp in $Depths) {
          $tbSubdir = "$BaseLogDir/ushcn/icfld_rcf/$DateTag/s${s}_ld${ld}_nh${nh}_eph${eph}_ed${ed}_dp${dp}"
          $tag = "ICFLD_USHCN_RCF_s${s}_ld${ld}_nh${nh}_eph${eph}_dp${dp}"
          $cmd = @(
            "python $RCFScript",
            "-d $Dataset -ot $Obs -bs $BatchSize -e $Epochs -es $EarlyStop",
            "-fn $Fn -ed $ed -ld $ld -nh $nh -dp $dp",
            "--cycle-length $CycleLength --time-max-hours $TimeMaxHours",
            "--gpu $GPU --tbon --logdir `"$tbSubdir`" --resume auto"
          ) -join " "
          Run-JobAt $tag $TrainDir $cmd
        }
      }
    }
  }
}

Write-Host ""
Write-Host "==> Sweep finished. Logs in $JobLogDir"
Write-Host "Start TensorBoard with:  tensorboard --logdir $BaseLogDir"
