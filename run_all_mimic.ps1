# run_all_mimic.ps1  — sanity run (exactly 5 epochs)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---- knobs ----
$GPU      = 0
$BS       = 32
$EPOCHS   = 70
$DATASET  = "mimic"
$LOGBASE  = "runs\mimic_sanity"
$TAIL_ERR = 60
$HEAD_ERR = 40

# Resolve to repo root
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptRoot

# CUDA visible device
$env:CUDA_VISIBLE_DEVICES = "$GPU"

# Logs
$DateTag   = Get-Date -Format "yyyyMMdd_HHmmss"
$JobLogDir = Join-Path $ScriptRoot ("joblogs\mimic_{0}" -f $DateTag)
New-Item -ItemType Directory -Force -Path $JobLogDir  | Out-Null
New-Item -ItemType Directory -Force -Path $LOGBASE    | Out-Null

# Python exe
$PythonExe = "python"
try { $null = Get-Command $PythonExe -ErrorAction Stop } catch { $PythonExe = "py" }

# Use .NET temp dir (avoids ${env:TEMP} quoting issues)
$TempDir = [System.IO.Path]::GetTempPath()

# ---------------- helpers ----------------
function Get-HelpText {
  param([string]$ScriptPath, [string]$WorkDir)

  $base   = [System.IO.Path]::GetFileNameWithoutExtension($ScriptPath)
  $outPath = Join-Path $TempDir ("help_{0}.out.txt" -f $base)
  $errPath = Join-Path $TempDir ("help_{0}.err.txt" -f $base)

  $psi = @{
    FilePath = $PythonExe
    ArgumentList = @($ScriptPath, "--help")
    WorkingDirectory = $WorkDir
    NoNewWindow = $true
    Wait = $true
    RedirectStandardOutput = $outPath
    RedirectStandardError  = $errPath
  }
  try {
    $p = Start-Process @psi -PassThru
    if (Test-Path $outPath) { return (Get-Content -Path $outPath -Raw) } else { return "" }
  } catch { return "" }
}

function Find-Flag {
  param([string]$HelpText, [string[]]$Candidates)
  foreach ($c in $Candidates) { if ($HelpText -match [Regex]::Escape($c)) { return $c } }
  return $null
}

function Start-ModelJob {
  param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][string]$ScriptPath
  )

  if (-not (Test-Path $ScriptPath)) {
    Write-Warning ("Skipping {0} — script not found: {1}" -f $Name,$ScriptPath)
    return
  }

  $WorkDir   = Split-Path -Parent $ScriptPath
  $HelpText  = Get-HelpText -ScriptPath $ScriptPath -WorkDir $WorkDir

  $flagDataset = Find-Flag -HelpText $HelpText -Candidates @("--dataset","--dataset_name","--data","--dset")
  $flagEpochs  = Find-Flag -HelpText $HelpText -Candidates @("--epochs","--num_epochs","--max_epochs")
  $flagBS      = Find-Flag -HelpText $HelpText -Candidates @("--batch_size","--batchsize","--bs","-b")
  $flagGPU     = Find-Flag -HelpText $HelpText -Candidates @("--gpu","--device","--cuda","--gpus")

  $args = @($ScriptPath)
  if ($flagDataset) { $args += @($flagDataset, $DATASET) }
  if ($flagEpochs)  { $args += @($flagEpochs,  $EPOCHS.ToString()) }
  if ($flagBS)      { $args += @($flagBS,      $BS.ToString()) }
  if ($flagGPU)     { $args += @($flagGPU,     $GPU.ToString()) }

  $outLog = Join-Path $JobLogDir ("{0}_{1}.out.log" -f $Name,$DateTag)
  $errLog = Join-Path $JobLogDir ("{0}_{1}.err.log" -f $Name,$DateTag)

  Write-Host "------------------------------------------------------------"
  Write-Host (" MODEL  : {0}" -f $Name)
  Write-Host (" CWD    : {0}" -f $WorkDir)
  Write-Host (" SCRIPT : {0}" -f $ScriptPath)
  Write-Host (" ARGS   : {0}" -f ($args -join ' '))
  Write-Host (" LOGOUT : {0}" -f $outLog)
  Write-Host (" LOGERR : {0}" -f $errLog)
  Write-Host (" START  : {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
  Write-Host "------------------------------------------------------------"

  $psi = @{
    FilePath = $PythonExe
    ArgumentList = $args
    WorkingDirectory = $WorkDir
    NoNewWindow = $true
    Wait = $true
    RedirectStandardOutput = $outLog
    RedirectStandardError  = $errLog
  }

  $p = Start-Process @psi -PassThru
  $exit = $p.ExitCode
  if ($exit -ne 0) {
    Write-Warning ("[{0}] exited with code {1}" -f $Name,$exit)
    if (Test-Path $errLog) {
      Write-Host ("----- {0}: first {1} lines of STDERR -----" -f $Name,$HEAD_ERR) -ForegroundColor Yellow
      Get-Content $errLog -TotalCount $HEAD_ERR | ForEach-Object { $_ }
      Write-Host ("----- {0}: last  {1} lines of STDERR -----" -f $Name,$TAIL_ERR) -ForegroundColor Yellow
      Get-Content $errLog | Select-Object -Last $TAIL_ERR | ForEach-Object { $_ }
    }
  } else {
    Write-Host ("[{0}] completed successfully." -f $Name)
  }
}

# ---- your exact training script paths ----
$MODELS = @(
  @{ Name = "FLD";       Script = (Join-Path $ScriptRoot "FLD\train_FLD.py") },
  @{ Name = "IC-FLD";    Script = (Join-Path $ScriptRoot "FLD_ICC\train_FLD_ICC.py") },
  @{ Name = "MTAN";      Script = (Join-Path $ScriptRoot "mtan\train_mtan.py") },
  @{ Name = "Grafiti";   Script = (Join-Path $ScriptRoot "Grafiti\train_grafiti.py") },
  @{ Name = "tPatchGNN"; Script = (Join-Path $ScriptRoot "tPatchGNN\run_models.py") }
)

Write-Host "======= MIMIC sanity run =======" -ForegroundColor Green
Write-Host ("GPU           : {0}" -f $GPU)
Write-Host ("Batch size    : {0}" -f $BS)
Write-Host ("Epochs        : {0}" -f $EPOCHS)
Write-Host ("Log base      : {0}" -f $LOGBASE)
Write-Host ("Job log dir   : {0}" -f $JobLogDir)
Write-Host "=================================" -ForegroundColor Green

foreach ($m in $MODELS) {
  Start-ModelJob -Name $m.Name -ScriptPath $m.Script
}

Write-Host "All MIMIC sanity jobs finished."
