$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

python scripts\benchmark_forward_runtime.py `
  --model_dir weights\23-36-37\model_best_bp2_serialize.pth `
  --height 480 `
  --width 640 `
  --valid_iters 4 `
  --warmup 10 `
  --repeat 30 `
  --optimize_build_volume triton `
  --out_dir live_demo_code_20260628\outputs\runtime_base_v4 `
  --tag live_base_v4

$RepGeo = "output_eval\stage29_repgeo_gamma_sweep_20260617\checkpoints\repgeo_gamma_1.pth"
if (Test-Path $RepGeo) {
  python scripts\benchmark_forward_runtime.py `
    --model_dir $RepGeo `
    --height 480 `
    --width 640 `
    --valid_iters 4 `
    --warmup 10 `
    --repeat 30 `
    --optimize_build_volume triton `
    --out_dir live_demo_code_20260628\outputs\runtime_repgeo_v4 `
    --tag live_repgeo_v4
} else {
  Write-Host "RepGeo compiled checkpoint not found, base runtime is already measured."
}


