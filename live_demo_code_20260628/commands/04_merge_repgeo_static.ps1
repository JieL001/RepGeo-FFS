$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
. (Join-Path $PSScriptRoot "_select_python.ps1")

$State = "output_eval\remote_stage23_repgeo_seed2_long_20260617\repgeo_ffs_repgeo_seed2_long_20260617_090506\repgeo_student\model_best_epe.pth"
$Base = "weights\23-36-37\model_best_bp2_serialize.pth"

if (!(Test-Path $State)) {
  Write-Host "RepGeo training-state checkpoint not found, skip merge demo: $State"
  exit 0
}

& $DemoPython scripts\merge_lora_for_inference.py `
  --model_dir $State `
  --base_model_dir $Base `
  --out_model live_demo_code_20260628\outputs\repgeo_merged_demo.pth `
  --report_json live_demo_code_20260628\outputs\repgeo_merged_demo_report.json `
  --lora_rank 4 `
  --lora_alpha 8 `
  --lora_targets cost update upsample `
  --dense_grouped

Write-Host "Merged static checkpoint: live_demo_code_20260628\outputs\repgeo_merged_demo.pth"
Write-Host "Merge report: live_demo_code_20260628\outputs\repgeo_merged_demo_report.json"
