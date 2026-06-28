# 权重与现场演示路线

本项目现场应展示两条线：

1. **原文 Fast-FoundationStereo 复现线**：使用原论文发布权重，演示原模型代码能跑通。
2. **我们的 RepGeo-FFS 改进线**：使用我们训练/折叠后的静态权重，演示改进模型也能直接推理、评估和导出。

## 1. 权重清单

| 权重 | 路径 | 用途 | 是否放入 zip |
| --- | --- | --- | --- |
| FFS base 原文权重 | `weights/23-36-37/model_best_bp2_serialize.pth` | 原文模型推理、复现评估 | 不放入，工程中已有 |
| RepGeo-Target `gamma=1.0` | `output_eval/stage29_repgeo_gamma_sweep_20260617/checkpoints/repgeo_gamma_1.pth` | 我们的 KITTI 目标域强适配静态模型 | 不放入，工程中已有 |
| RepGeo-Calibrated `gamma=0.01` | `output_eval/stage29_repgeo_gamma_sweep_20260617/checkpoints_fine/repgeo_gamma_0p01.pth` | 我们的跨域保守校准静态模型 | 不放入，工程中已有 |
| RepGeo 训练态 checkpoint | `output_eval/remote_stage23_repgeo_seed2_long_20260617/repgeo_ffs_repgeo_seed2_long_20260617_090506/repgeo_student/model_best_epe.pth` | 展示从 wrapper 到静态模型的折叠过程 | 不放入，工程中已有 |

zip 包不内嵌这些 `.pth`，因为每个约 68MB。现场在同一台电脑演示时直接读取工程目录中的权重即可。

## 2. 原文 FFS 复现演示

### 2.1 原文权重推理

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\01_inference_no_window.ps1
```

输出：

```text
live_demo_code_20260628/outputs/infer_base/infer_board.png
```

讲法：

> 这一步使用原文 FFS base 权重，证明原模型推理路径可以跑通。

### 2.2 原文权重本地评估 smoke

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\02_eval_only_smoke.ps1
```

讲法：

> 这一步使用 KITTI 小样本验证，证明原模型的指标计算路径可以跑通。

## 3. 我们的 RepGeo-FFS 改进演示

### 3.1 RepGeo-Target 推理

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\06_inference_repgeo_target.ps1
```

输出：

```text
live_demo_code_20260628/outputs/infer_repgeo_target_gamma1/infer_board.png
```

讲法：

> 这一步使用我们导出的 RepGeo-Target `gamma=1.0` 静态权重，展示目标域强适配模型可以像原 FFS 一样直接前向推理。

### 3.2 RepGeo-Calibrated 推理

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\07_inference_repgeo_calibrated.ps1
```

输出：

```text
live_demo_code_20260628/outputs/infer_repgeo_calibrated_gamma0p01/infer_board.png
```

讲法：

> 这一步使用 `gamma=0.01` 保守校准模型，表示在跨域场景中可以降低校准强度。

### 3.3 FFS base 与 RepGeo target 小样本对照评估

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\08_eval_base_vs_repgeo_smoke.ps1
```

输出：

```text
live_demo_code_20260628/outputs/eval_smoke_base.json
live_demo_code_20260628/outputs/eval_smoke_repgeo_target.json
```

讲法：

> 这不是完整结果表，只是现场小样本 smoke test。正式指标以报告中的完整 KITTI local-val 和 official private-preview 为准。

## 4. 折叠过程演示

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\04_merge_repgeo_static.ps1
```

预期输出：

```text
before merge: lora_modules=72
after merge: lora_modules=0
```

讲法：

> 这一步说明我们的最终 RepGeo-FFS 不是推理时外挂 LoRA，而是把训练期残差折叠回 FFS 的普通卷积权重。

## 5. 推荐现场顺序

如果时间只有 5 分钟：

1. `00_check_demo_assets.ps1`
2. `01_inference_no_window.ps1`
3. `06_inference_repgeo_target.ps1`
4. 打开 `WEIGHTS_AND_RUNS_权重与演示路线.md` 说明权重对应关系
5. 打开 `CODE_WALKTHROUGH_现场讲解.md` 指给老师看损失函数和折叠实现


