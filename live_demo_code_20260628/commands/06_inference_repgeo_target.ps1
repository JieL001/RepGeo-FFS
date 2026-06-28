$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

$Model = "output_eval\stage29_repgeo_gamma_sweep_20260617\checkpoints\repgeo_gamma_1.pth"
if (!(Test-Path $Model)) {
  throw "Missing RepGeo target checkpoint: $Model"
}

python live_demo_code_20260628\commands\infer_no_window.py `
  --model_dir $Model `
  --left_file demo_data\left.png `
  --right_file demo_data\right.png `
  --out_dir live_demo_code_20260628\outputs\infer_repgeo_target_gamma1 `
  --valid_iters 4 `
  --optimize_build_volume pytorch1

Write-Host "Open result: live_demo_code_20260628\outputs\infer_repgeo_target_gamma1\infer_board.png"


