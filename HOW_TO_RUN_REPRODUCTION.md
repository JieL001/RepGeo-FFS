# How to Run Reproduction and Live Inspection

This guide is for classroom inspection. Commands are intended to run from the project root on Windows PowerShell.

```powershell
cd C:\Users\PC\Desktop\dev\Fast-FoundationStereo-master
conda activate ffs
```

The live scripts now use `F:\miniforge3\envs\ffs\python.exe` automatically when it exists, so they do not accidentally call a CPU-only system Python.

## 1. Required Files

Original Fast-FoundationStereo checkpoint:

```text
weights/23-36-37/model_best_bp2_serialize.pth
```

KITTI data for evaluation/training smoke tests:

```text
data_scene_flow/training/image_2
data_scene_flow/training/image_3
data_scene_flow/training/disp_occ_0
data_scene_flow/training/disp_noc_0
```

Optional RepGeo checkpoints:

```text
output_eval/stage29_repgeo_gamma_sweep_20260617/checkpoints/repgeo_gamma_1.pth
output_eval/stage29_repgeo_gamma_sweep_20260617/checkpoints_fine/repgeo_gamma_0p01.pth
```

## 2. Environment Check

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\00_check_demo_assets.ps1
```

This checks CUDA, Torch, model weights, demo images, KITTI folders, and RepGeo checkpoints.

## 3. Original FFS Inference

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\01_inference_no_window.ps1
```

Outputs:

```text
live_demo_code_20260628/outputs/infer_base/infer_board.png
live_demo_code_20260628/outputs/infer_base/disp_vis.png
live_demo_code_20260628/outputs/boards/live_inference_comparison.png
```

## 4. Multi-Sample Live Inference Board

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\10_multi_inference_board.ps1
```

Output:

```text
live_demo_code_20260628/outputs/boards/multi_sample_live_inference.png
```

This board runs real forward passes on multiple stereo pairs and compares FFS base, RepGeo target, and RepGeo calibrated checkpoints.

## 5. Evaluation Smoke Test

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\08_eval_base_vs_repgeo_smoke.ps1
```

Outputs:

```text
live_demo_code_20260628/outputs/eval_smoke_base.json
live_demo_code_20260628/outputs/eval_smoke_repgeo_target.json
live_demo_code_20260628/outputs/boards/eval_metrics_board.png
```

The PNG board summarizes EPE, D1, bad3, visible EPE, and occluded EPE without relying on terminal text.

## 6. RepGeo Training Smoke Test

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\03_train_smoke_repgeo.ps1
```

Outputs:

```text
live_demo_code_20260628/outputs/train_smoke_repgeo/train_smoke_repgeo.log
live_demo_code_20260628/outputs/boards/train_smoke_board.png
```

This short run uses 4 training samples, 2 validation samples, and 1 epoch. It verifies:

- data loading,
- supervised KITTI loss,
- boundary/visibility loss,
- frozen FFS prior preservation,
- RepGeo sparse and delta regularization,
- backward pass,
- checkpoint writing.

## 7. Static RepGeo Folding

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\04_merge_repgeo_static.ps1
```

Conceptually:

```text
training:   y = Conv(x, W0) + alpha * scaling * Conv_up(Conv_down(x))
deployment: W* = W0 + alpha * scaling * DeltaW
```

After folding, the deployed checkpoint is FFS-style: no LoRA wrapper, no router, and no input-dependent branch.

## 8. Runtime Benchmark

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\05_runtime_3090_style.ps1
```

Runtime depends on GPU. Use numbers only when comparing methods under the same hardware and script.

## 9. Full Training Command

Print the full command without starting a long run:

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\09_train_full_reference.ps1
```

Start the full reference run:

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\09_train_full_reference.ps1 -Run
```

For classroom inspection, the smoke test is usually enough to prove that the training code path is executable.
