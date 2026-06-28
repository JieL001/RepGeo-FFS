$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
. (Join-Path $PSScriptRoot "_select_python.ps1")

$OutDir = "live_demo_code_20260628\outputs\train_smoke_repgeo"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Log = Join-Path $OutDir "train_smoke_repgeo.log"
$Stdout = Join-Path $OutDir "train_smoke_repgeo.stdout.log"
$Stderr = Join-Path $OutDir "train_smoke_repgeo.stderr.log"

$TrainArgs = @(
  "scripts\train_kitti.py",
  "--data_root", "data_scene_flow",
  "--out_dir", $OutDir,
  "--epochs", "1",
  "--batch_size", "1",
  "--num_workers", "0",
  "--max_train_samples", "4",
  "--max_val_samples", "2",
  "--crop_h", "256",
  "--crop_w", "768",
  "--valid_iters", "2",
  "--lora_adapt",
  "--lora_rank", "4",
  "--lora_alpha", "8",
  "--lora_dropout", "0.05",
  "--visibility_supervision",
  "--boundary_loss_weight", "0.5",
  "--preserve_model_dir", "weights\23-36-37\model_best_bp2_serialize.pth",
  "--preserve_loss_weight", "0.05",
  "--preserve_high_conf_prior",
  "--repgeo_sparse_weight", "0.00001",
  "--repgeo_delta_weight", "0.0001"
)

Remove-Item -Force -ErrorAction SilentlyContinue $Stdout, $Stderr, $Log
$Proc = Start-Process -FilePath $DemoPython -ArgumentList $TrainArgs -Wait -PassThru -NoNewWindow -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr

$Merged = @()
if (Test-Path $Stdout) { $Merged += Get-Content $Stdout }
if (Test-Path $Stderr) { $Merged += Get-Content $Stderr }
$Merged | Set-Content -Path $Log -Encoding UTF8
$Merged | ForEach-Object { Write-Host $_ }

if ($Proc.ExitCode -ne 0) {
  exit $Proc.ExitCode
}

& $DemoPython live_demo_code_20260628\commands\make_demo_boards.py --mode train

Write-Host "Train smoke output: live_demo_code_20260628\outputs\train_smoke_repgeo"
Write-Host "Train log         : live_demo_code_20260628\outputs\train_smoke_repgeo\train_smoke_repgeo.log"
Write-Host "Train board       : live_demo_code_20260628\outputs\boards\train_smoke_board.png"
