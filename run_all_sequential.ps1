# run_all_sequential.ps1 — run your four PS1 jobs one by one
# Files expected in the same folder:
#   - run_all_ushcn.ps1
#   - run_all_physionet.ps1
#   - run_activity_sanity.ps1
#   - run_all_mimic.ps1
#
# Usage:
#   .\run_all_sequential.ps1              # stop on first error
#   .\run_all_sequential.ps1 -ContinueOnError  # keep going if one fails

param(
  [switch]$ContinueOnError
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Work from this script's directory (repo root)
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptRoot

# Make sure no allocator tweaks leak into child runs (per your request)
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue
[Environment]::SetEnvironmentVariable("PYTORCH_CUDA_ALLOC_CONF",$null,"Process")

# If you must ensure a specific conda env, uncomment and set the path:
# $env:PATH = "C:\Users\yazda\Miniconda3\envs\thesis;$env:PATH"

$Steps = @(
  "run_all_ushcn.ps1",
  "run_all_physionet.ps1",
  "run_activity_sanity.ps1",
  "run_all_mimic.ps1"
)

function Invoke-Step {
  param([string]$File)
  if (-not (Test-Path $File)) {
    $msg = "Missing script: $File"
    if ($ContinueOnError) { Write-Warning $msg; return }
    else { throw $msg }
  }

  Write-Host "============================================================"
  Write-Host "Running: $File"
  Write-Host "Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
  Write-Host "============================================================"

  # Spawn a child PowerShell so 'exit' inside the script doesn't kill this wrapper.
  $psBin = (Get-Process -Id $PID).Path   # current pwsh/powershell executable
  $args  = @("-NoProfile","-ExecutionPolicy","Bypass","-File",$File)

  $p = Start-Process -FilePath $psBin -ArgumentList $args -PassThru -Wait
  $code = $p.ExitCode

  if ($code -ne 0) {
    $msg = "$File exited with code $code"
    if ($ContinueOnError) { Write-Warning $msg }
    else { throw $msg }
  } else {
    Write-Host "$File completed successfully." -ForegroundColor Green
  }
}

foreach ($f in $Steps) { Invoke-Step $f }

Write-Host "All scripts finished." -ForegroundColor Cyan
