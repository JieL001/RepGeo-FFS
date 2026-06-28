$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
. (Join-Path $PSScriptRoot "_select_python.ps1")

$Base = "weights\23-36-37\model_best_bp2_serialize.pth"
$RepGeo = "output_eval\stage29_repgeo_gamma_sweep_20260617\checkpoints\repgeo_gamma_1.pth"

Write-Host "---- FFS base eval smoke ----"
& $DemoPython scripts\train_kitti.py `
  --model_dir $Base `
  --data_root data_scene_flow `
  --eval_only `
  --max_val_samples 2 `
  --valid_iters 2 `
  --num_workers 0 `
  --report_json live_demo_code_20260628\outputs\eval_smoke_base.json

Write-Host "---- RepGeo target gamma=1.0 eval smoke ----"
& $DemoPython scripts\train_kitti.py `
  --model_dir $RepGeo `
  --data_root data_scene_flow `
  --eval_only `
  --max_val_samples 2 `
  --valid_iters 2 `
  --num_workers 0 `
  --report_json live_demo_code_20260628\outputs\eval_smoke_repgeo_target.json

& $DemoPython live_demo_code_20260628\commands\make_demo_boards.py --mode eval

Write-Host "Reports:"
Write-Host "  live_demo_code_20260628\outputs\eval_smoke_base.json"
Write-Host "  live_demo_code_20260628\outputs\eval_smoke_repgeo_target.json"
Write-Host "Board:"
Write-Host "  live_demo_code_20260628\outputs\boards\eval_metrics_board.png"
