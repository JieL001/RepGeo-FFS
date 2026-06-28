param(
  [switch]$Run
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

$RunId = "live_full_repgeo_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$OutDir = "output_train\$RunId"

$ArgsList = @(
  "scripts\train_kitti.py",
  "--model_dir", "weights\23-36-37\model_best_bp2_serialize.pth",
  "--data_root", "data_scene_flow",
  "--out_dir", $OutDir,
  "--epochs", "8",
  "--batch_size", "1",
  "--num_workers", "2",
  "--crop_h", "320",
  "--crop_w", "960",
  "--valid_iters", "4",
  "--lora_adapt",
  "--lora_rank", "4",
  "--lora_alpha", "8",
  "--lora_dropout", "0.05",
  "--lora_targets", "cost", "update", "upsample",
  "--visibility_supervision",
  "--boundary_loss_weight", "0.5",
  "--preserve_model_dir", "weights\23-36-37\model_best_bp2_serialize.pth",
  "--preserve_loss_weight", "0.05",
  "--preserve_high_conf_prior",
  "--repgeo_sparse_weight", "0.00001",
  "--repgeo_delta_weight", "0.0001",
  "--seed", "0"
)

Write-Host "Full RepGeo training command:"
Write-Host "python $($ArgsList -join ' ')"
Write-Host ""
Write-Host "Output directory will be: $OutDir"
Write-Host ""

if (-not $Run) {
  Write-Host "This reference script only prints the full command by default."
  Write-Host "To really start full training, run:"
  Write-Host "powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\09_train_full_reference.ps1 -Run"
  exit 0
}

python @ArgsList


