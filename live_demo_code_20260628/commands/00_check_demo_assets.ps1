$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
. (Join-Path $PSScriptRoot "_select_python.ps1")

Write-Host "ROOT = $Root"
Write-Host "---- required files / folders ----"

$required = @(
  "weights\23-36-37\model_best_bp2_serialize.pth",
  "demo_data\left.png",
  "demo_data\right.png",
  "scripts\train_kitti.py",
  "scripts\merge_lora_for_inference.py",
  "scripts\benchmark_forward_runtime.py",
  "core\foundation_stereo.py",
  "core\lora.py",
  "core\datasets.py"
)

$optional = @(
  "data_scene_flow\training\image_2",
  "data_scene_flow\training\image_3",
  "data_scene_flow\training\disp_occ_0",
  "data_scene_flow\training\disp_noc_0",
  "output_eval\stage29_repgeo_gamma_sweep_20260617\checkpoints\repgeo_gamma_1.pth",
  "output_eval\stage29_repgeo_gamma_sweep_20260617\checkpoints_fine\repgeo_gamma_0p01.pth",
  "output_eval\remote_stage23_repgeo_seed2_long_20260617\repgeo_ffs_repgeo_seed2_long_20260617_090506\repgeo_student\model_best_epe.pth"
)

foreach ($p in $required) {
  if (Test-Path $p) {
    Write-Host "[OK]       $p"
  } else {
    Write-Host "[MISSING]  $p"
    throw "Required path missing: $p"
  }
}

Write-Host "---- optional experiment assets ----"
foreach ($p in $optional) {
  if (Test-Path $p) {
    Write-Host "[OK]       $p"
  } else {
    Write-Host "[OPTIONAL] $p"
  }
}

Write-Host "---- python / torch ----"
& $DemoPython -c "import sys, torch; print('python=', sys.version.split()[0]); print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda_name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
