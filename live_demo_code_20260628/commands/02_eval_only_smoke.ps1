$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

python scripts\train_kitti.py `
  --data_root data_scene_flow `
  --eval_only `
  --max_val_samples 2 `
  --valid_iters 2 `
  --num_workers 0 `
  --report_json live_demo_code_20260628\outputs\eval_smoke.json

Write-Host "Eval report: live_demo_code_20260628\outputs\eval_smoke.json"


