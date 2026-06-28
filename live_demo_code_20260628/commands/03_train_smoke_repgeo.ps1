$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

python scripts\train_kitti.py `
  --data_root data_scene_flow `
  --out_dir live_demo_code_20260628\outputs\train_smoke_repgeo `
  --epochs 1 `
  --batch_size 1 `
  --num_workers 0 `
  --max_train_samples 4 `
  --max_val_samples 2 `
  --crop_h 256 `
  --crop_w 768 `
  --valid_iters 2 `
  --lora_adapt `
  --lora_rank 4 `
  --lora_alpha 8 `
  --lora_dropout 0.05 `
  --visibility_supervision `
  --boundary_loss_weight 0.5 `
  --preserve_model_dir weights\23-36-37\model_best_bp2_serialize.pth `
  --preserve_loss_weight 0.05 `
  --preserve_high_conf_prior `
  --repgeo_sparse_weight 0.00001 `
  --repgeo_delta_weight 0.0001

Write-Host "Train smoke output: live_demo_code_20260628\outputs\train_smoke_repgeo"


