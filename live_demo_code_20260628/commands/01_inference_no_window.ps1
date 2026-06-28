$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
. (Join-Path $PSScriptRoot "_select_python.ps1")

& $DemoPython live_demo_code_20260628\commands\infer_no_window.py `
  --model_dir weights\23-36-37\model_best_bp2_serialize.pth `
  --left_file demo_data\left.png `
  --right_file demo_data\right.png `
  --out_dir live_demo_code_20260628\outputs\infer_base `
  --valid_iters 4 `
  --optimize_build_volume pytorch1

& $DemoPython live_demo_code_20260628\commands\make_demo_boards.py --mode infer

Write-Host "Open result: live_demo_code_20260628\outputs\infer_base\infer_board.png"
Write-Host "Open board : live_demo_code_20260628\outputs\boards\live_inference_comparison.png"
