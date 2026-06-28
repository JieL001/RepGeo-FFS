$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
. (Join-Path $PSScriptRoot "_select_python.ps1")

Write-Host "=== Multi-sample live inference visualization ==="
Write-Host "This runs real forward passes on demo/KITTI samples and builds a clean comparison board."

& $DemoPython live_demo_code_20260628\commands\multi_infer_board.py `
  --max_samples 3 `
  --valid_iters 2 `
  --max_disp 192 `
  --precision fp32 `
  --optimize_build_volume pytorch1

if ($LASTEXITCODE -ne 0) {
  throw "multi_infer_board.py failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Open visualization:"
Write-Host "  live_demo_code_20260628\outputs\boards\multi_sample_live_inference.png"
