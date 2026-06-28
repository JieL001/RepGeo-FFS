# 如何运行复现与改进代码

本文档用于现场验收或本地复现实验流程。命令默认在 Windows PowerShell 中执行，并假设当前目录是项目根目录。

```powershell
cd C:\Users\PC\Desktop\dev\Fast-FoundationStereo-master
conda activate ffs
```

如果是从 GitHub 仓库重新下载代码，需要先准备原始 FFS 权重和 KITTI 数据。大文件没有放入 GitHub。

## 1. 必要文件

### 原文 FFS 权重

放到：

```text
weights/23-36-37/model_best_bp2_serialize.pth
```

### KITTI 数据目录

用于评估和训练 smoke test：

```text
data_scene_flow/training/image_2
data_scene_flow/training/image_3
data_scene_flow/training/disp_occ_0
data_scene_flow/training/disp_noc_0
```

### RepGeo 权重

如果要演示我们训练后的模型，放到：

```text
output_eval/stage29_repgeo_gamma_sweep_20260617/checkpoints/repgeo_gamma_1.pth
output_eval/stage29_repgeo_gamma_sweep_20260617/checkpoints_fine/repgeo_gamma_0p01.pth
```

其中：

- `repgeo_gamma_1.pth` 是 KITTI 目标域强适配模型；
- `repgeo_gamma_0p01.pth` 是跨域保守校准模型。

## 2. 环境与资源检查

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\00_check_demo_assets.ps1
```

这一步检查：

- CUDA 是否可用；
- 原文权重是否存在；
- demo 图像是否存在；
- 训练、评估、RepGeo 相关脚本是否存在；
- KITTI 目录是否完整。

## 3. 原文 Fast-FoundationStereo 推理复现

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\01_inference_no_window.ps1
```

输出位置：

```text
live_demo_code_20260628/outputs/infer_base/infer_board.png
live_demo_code_20260628/outputs/infer_base/disp_vis.png
live_demo_code_20260628/outputs/infer_base/disp.npy
```

这一步证明：原论文 FFS 权重可以完成从左右图输入到视差图输出的推理流程。

## 4. 原文 FFS 小样本评估复现

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\02_eval_only_smoke.ps1
```

输出位置：

```text
live_demo_code_20260628/outputs/eval_smoke.json
```

这一步只跑少量 KITTI 验证样本，用于证明数据读取、模型前向、EPE/D1/bad3 指标计算路径完整。正式结果以报告中的完整 local-val 和 official private-preview 为准。

## 5. RepGeo-FFS 推理演示

目标域强适配版本：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\06_inference_repgeo_target.ps1
```

跨域保守校准版本：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\07_inference_repgeo_calibrated.ps1
```

输出位置：

```text
live_demo_code_20260628/outputs/infer_repgeo_target
live_demo_code_20260628/outputs/infer_repgeo_calibrated
```

这一步证明：我们训练并折叠后的 RepGeo checkpoint 可以按普通 FFS 推理流程运行。

## 6. FFS base 与 RepGeo 小样本对比评估

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\08_eval_base_vs_repgeo_smoke.ps1
```

输出位置：

```text
live_demo_code_20260628/outputs/eval_smoke_base.json
live_demo_code_20260628/outputs/eval_smoke_repgeo_target.json
```

这一步证明：同一套评估脚本可以分别加载原文 FFS base 和 RepGeo 改进模型。

## 7. RepGeo 训练 smoke test

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\03_train_smoke_repgeo.ps1
```

该脚本只使用少量样本和 1 个 epoch，用于现场证明训练链路可以跑通。它包含：

- KITTI 监督损失；
- boundary / visibility 相关监督；
- frozen FFS prior preservation；
- RepGeo alpha 稀疏约束；
- RepGeo 残差幅度约束；
- 反向传播和参数更新。

输出位置：

```text
live_demo_code_20260628/outputs/train_smoke_repgeo
```

## 8. 完整训练命令

只打印完整训练命令，不启动训练：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\09_train_full_reference.ps1
```

真正启动完整训练：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\09_train_full_reference.ps1 -Run
```

完整训练输出目录类似：

```text
output_train/live_full_repgeo_YYYYMMDD_HHMMSS
```

现场一般不建议跑完整训练，因为需要较长 GPU 时间；验收时跑 smoke test 即可证明训练代码路径。

## 9. RepGeo 静态折叠导出

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\04_merge_repgeo_static.ps1
```

该步骤展示 RepGeo 的核心部署形式：

```text
训练时：y = Conv(x, W0) + alpha * scaling * Conv_up(Conv_down(x))
部署时：W* = W0 + alpha * scaling * DeltaW
```

折叠后模型不再保留：

- LoRA wrapper；
- RepGeo wrapper；
- router；
- input-dependent if-else 分支。

最终是普通 FFS-style checkpoint。

## 10. Runtime 测试

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\05_runtime_3090_style.ps1
```

该脚本按报告中的 RTX 3090 口径测试 FFS / RepGeo 前向耗时。不同 GPU 上数字会变化，因此报告中只使用同一硬件下的相对比较。

## 11. 现场讲解时重点打开的源码

| 目的 | 文件 |
| --- | --- |
| FFS 主网络前向 | `core/foundation_stereo.py` |
| RepGeo / Conv-LoRA 单元 | `core/lora.py` |
| KITTI 训练入口 | `scripts/train_kitti.py` |
| 损失函数实现 | `scripts/train_kitti.py` |
| 静态折叠导出 | `scripts/merge_lora_for_inference.py` |
| RepGeo gamma 导出 | `scripts/make_repgeo_gamma_checkpoint.py` |
| 无窗口推理脚本 | `live_demo_code_20260628/commands/infer_no_window.py` |

## 12. 推荐现场演示顺序

如果时间只有 3 到 5 分钟，按这个顺序：

1. 运行资源检查：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\00_check_demo_assets.ps1
```

2. 跑原文 FFS 推理：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\01_inference_no_window.ps1
```

3. 跑训练 smoke：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\03_train_smoke_repgeo.ps1
```

4. 展示静态折叠：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\04_merge_repgeo_static.ps1
```

5. 如果老师要求看我们训练后的模型，再跑：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\06_inference_repgeo_target.ps1
```

