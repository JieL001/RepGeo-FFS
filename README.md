# RepGeo-FFS

Course project code for reproducing **Fast-FoundationStereo** and implementing two adaptation extensions:

1. **FFSOmega-RGF**: a reliability-replay and validity-boundary adaptation baseline.
2. **RepGeo-FFS**: a re-parameterizable geometric residual calibration method that folds training-time residual units back into a static FFS-style checkpoint.

This repository is prepared for course inspection and code demonstration. Large datasets, trained checkpoints, and generated experiment outputs are not included.

## What Is Included

```text
core/                         model code, FFS backbone, RepGeo/LoRA residual units
scripts/                      training, evaluation, inference, merging, runtime scripts
demo_data/                    small demo stereo pair
live_demo_code_20260628/      classroom/live inspection scripts and explanations
docker/                       original environment files
assets/                       figures used by original FFS documentation
README_FFS_ORIGINAL.md        original Fast-FoundationStereo README snapshot
```

Key files for inspection:

| Purpose | File |
| --- | --- |
| FFS model forward | `core/foundation_stereo.py` |
| RepGeo / Conv-LoRA residual unit | `core/lora.py` |
| KITTI training and losses | `scripts/train_kitti.py` |
| Static residual folding | `scripts/merge_lora_for_inference.py` |
| RepGeo gamma checkpoint export | `scripts/make_repgeo_gamma_checkpoint.py` |
| Inference demo | `scripts/run_demo.py` and `live_demo_code_20260628/commands/infer_no_window.py` |
| How to run reproduction | `HOW_TO_RUN_REPRODUCTION.md` |
| Live inspection guide | `live_demo_code_20260628/00_快速上手指南.md` |
| Teacher checklist | `live_demo_code_20260628/TEACHER_CHECKLIST_验收问答准备.md` |

## Environment

```bash
conda create -n ffs python=3.12
conda activate ffs
pip install torch==2.6.0 torchvision==0.21.0 xformers --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The original FFS code also provides Docker files under `docker/`.

## Required External Files

Large files are intentionally excluded from GitHub.

### 1. Original FFS Checkpoint

Place the released Fast-FoundationStereo checkpoint at:

```text
weights/23-36-37/model_best_bp2_serialize.pth
```

### 2. RepGeo-FFS Checkpoints

For local course demonstration, the following trained checkpoints were used:

```text
output_eval/stage29_repgeo_gamma_sweep_20260617/checkpoints/repgeo_gamma_1.pth
output_eval/stage29_repgeo_gamma_sweep_20260617/checkpoints_fine/repgeo_gamma_0p01.pth
```

They are not stored in this repository because each checkpoint is large. The code for training and exporting them is included.

### 3. KITTI Training Data

For evaluation/training smoke tests, prepare:

```text
data_scene_flow/training/image_2
data_scene_flow/training/image_3
data_scene_flow/training/disp_occ_0
data_scene_flow/training/disp_noc_0
```

## Quick Live Demo

In PowerShell:

```powershell
conda activate ffs
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\00_check_demo_assets.ps1
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\01_inference_no_window.ps1
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\03_train_smoke_repgeo.ps1
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\04_merge_repgeo_static.ps1
```

If RepGeo checkpoints are available locally:

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\06_inference_repgeo_target.ps1
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\08_eval_base_vs_repgeo_smoke.ps1
```

## Qualitative Evidence

The following paper-style comparison boards are included for course inspection. They use fixed samples and consistent method columns to show target-domain gains and cross-domain behavior.

### KITTI target-domain qualitative gains

![KITTI target-domain qualitative gains](assets/qualitative/cvpr_long_kitti_zoom_and_methods.png)

### Multi-dataset comparison matrix

![Multi-dataset qualitative comparison](assets/qualitative/cvpr_long_multidataset_9x7_repgeo.png)

### ETH3D / Middlebury cross-domain comparison

![Cross-domain qualitative comparison](assets/qualitative/cvpr_long_crossdomain_eth3d_middlebury.png)

Additional long appendix board:

[assets/qualitative/cvpr_long_appendix_all_qualitative.png](assets/qualitative/cvpr_long_appendix_all_qualitative.png)

## Training

Short smoke training:

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\03_train_smoke_repgeo.ps1
```

Print the full training command:

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\09_train_full_reference.ps1
```

Actually start full training:

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\09_train_full_reference.ps1 -Run
```

The training objective is implemented in `scripts/train_kitti.py`:

```text
L = L_sequence + lambda_boundary L_boundary + lambda_prior L_prior
    + lambda_sparse |alpha| + lambda_delta ||DeltaW||
```

Core functions:

```text
sequence_loss()
visibility_aware_loss()
compute_preserve_loss()
repgeo_regularization()
train_one_epoch()
```

## RepGeo-FFS Concept

During training, selected FFS Conv2d/Conv3d layers are wrapped as:

```text
y = Conv(x, W0) + alpha * scaling * Conv_up(Conv_down(x))
```

During deployment, the residual is folded:

```text
W* = W0 + alpha * scaling * DeltaW
```

After folding, the checkpoint contains ordinary FFS-style weights and does not require LoRA wrappers, a router, or input-dependent branch selection.

## Official KITTI Evaluation Evidence

The RepGeo-FFS result was submitted to the official KITTI Scene Flow / Stereo 2015 test server. The private-preview/check page is:

[KITTI official private-preview result](https://www.cvlibs.net/datasets/kitti/user_submit_check_login.php?benchmark=scene_flow&user=aa376795597a4f18ec9c0747279d694da3811c7c&result=297134ba0b3f09b59d7e383ea230b0d2a3934f00)

This link is used as course inspection evidence that the result was evaluated by the KITTI server. It is not claimed as a public leaderboard ranking.

## Notes

- This repository focuses on code inspection and reproducibility.
- Official datasets and checkpoints must be downloaded separately.
- The original Fast-FoundationStereo implementation and license are preserved. See `README_FFS_ORIGINAL.md` and `LICENSE.txt`.


