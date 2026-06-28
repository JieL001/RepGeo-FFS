# 现场演示代码包：FFS 复现、FFSOmega-R、RepGeo-FFS

本目录用于课堂或答辩现场快速展示代码实现。它不替代原工程，而是把老师最可能检查的代码入口、损失函数、训练命令、推理命令和静态折叠命令集中到一个地方。

## 1. 推荐现场演示顺序

| 顺序 | 演示内容 | 命令或文件 | 说明 |
| --- | --- | --- | --- |
| 1 | 环境与资源检查 | `commands/00_check_demo_assets.ps1` | 检查权重、demo 图、训练脚本、CUDA |
| 2 | 网络前向 | `source_snapshot/core__foundation_stereo.py` | 展示双目输入到视差输出的主干 |
| 3 | RepGeo/LoRA 单元 | `source_snapshot/core__lora.py` | 展示残差单元、`repgeo_alpha`、低秩卷积 |
| 4 | 损失函数 | `source_snapshot/scripts__train_kitti.py` | 展示 supervised loss、replay loss、prior loss、RepGeo 正则 |
| 5 | 无窗口推理 | `commands/01_inference_no_window.ps1` | 生成 `infer_board.png`，不弹 GUI，不会卡住 |
| 6 | 我们的 RepGeo-Target 推理 | `commands/06_inference_repgeo_target.ps1` | 加载 `gamma=1.0` 改进权重 |
| 7 | 我们的 RepGeo-Calibrated 推理 | `commands/07_inference_repgeo_calibrated.ps1` | 加载 `gamma=0.01` 改进权重 |
| 8 | 原文 vs 改进评估 | `commands/08_eval_base_vs_repgeo_smoke.ps1` | 同一小样本上分别评估 FFS 和 RepGeo |
| 9 | 小样本评估 | `commands/02_eval_only_smoke.ps1` | 验证 eval-only 路径 |
| 10 | 小样本训练 | `commands/03_train_smoke_repgeo.ps1` | 只跑 4 个样本，证明训练和反传路径 |
| 11 | 完整训练命令展示 | `commands/09_train_full_reference.ps1` | 默认只打印完整训练命令，`-Run` 才启动 |
| 12 | 静态折叠 | `commands/04_merge_repgeo_static.ps1` | 把 RepGeo/LoRA wrapper 合并成 FFS-style checkpoint |
| 13 | Runtime | `commands/05_runtime_3090_style.ps1` | 按 480x640、v4、warmup/repeat 测速 |

现场建议只跑第 1、5、6 步；如果时间够，再跑第 7 或第 8 步。完整训练结果已经在实验报告中记录，现场不需要重新长训。

## 1.1 权重区分

| 演示对象 | 权重路径 | 对应命令 |
| --- | --- | --- |
| 原文 FFS base | `weights/23-36-37/model_best_bp2_serialize.pth` | `01_inference_no_window.ps1` |
| 我们的 RepGeo-Target | `output_eval/stage29_repgeo_gamma_sweep_20260617/checkpoints/repgeo_gamma_1.pth` | `06_inference_repgeo_target.ps1` |
| 我们的 RepGeo-Calibrated | `output_eval/stage29_repgeo_gamma_sweep_20260617/checkpoints_fine/repgeo_gamma_0p01.pth` | `07_inference_repgeo_calibrated.ps1` |

详细说明见 `WEIGHTS_AND_RUNS_权重与演示路线.md`。

训练启动和损失函数说明见 `TRAINING_GUIDE_训练启动与代码细节.md`。

## 2. 代码实现索引

| 问题 | 代码位置 | 说明 |
| --- | --- | --- |
| 模型输入输出在哪里？ | `source_snapshot/core__foundation_stereo.py` | `FastFoundationStereo.forward` 完成特征、代价体、迭代更新、上采样 |
| LoRA/RepGeo 改了哪里？ | `source_snapshot/core__lora.py` | `LoRAConv2d/LoRAConv3d` 包装 Conv2d/Conv3d，并引入 `repgeo_alpha` |
| 训练入口在哪里？ | `source_snapshot/scripts__train_kitti.py` | `main -> train_one_epoch -> validate` |
| KITTI 数据怎么读？ | `source_snapshot/core__datasets.py` | `KittiStereoDataset` 检查 `image_2/image_3/disp_occ_0/disp_noc_0` |
| supervised loss 在哪里？ | `source_snapshot/scripts__train_kitti.py::sequence_loss` | 多轮视差预测加权监督 |
| 可见性/边界损失在哪里？ | `source_snapshot/scripts__train_kitti.py::visibility_aware_loss` | 加强可见区域与边界区域的训练 |
| reliability replay 在哪里？ | `source_snapshot/scripts__train_kitti.py::compute_pseudo_self_distill_loss` | 用冻结 teacher 伪标签和可靠性权重做保持 |
| 几何先验保持在哪里？ | `source_snapshot/scripts__train_kitti.py::compute_preserve_loss` | 让学生在高置信区域不偏离 FFS base |
| RepGeo 正则在哪里？ | `source_snapshot/scripts__train_kitti.py::repgeo_regularization` | 稀疏化 `alpha` 并约束残差规模 |
| 静态折叠在哪里？ | `source_snapshot/scripts__merge_lora_for_inference.py` | `W* = W0 + alpha * B * A`，导出无 wrapper checkpoint |
| 推理脚本在哪里？ | `commands/infer_no_window.py` | 读取左右图，输出视差可视化与 `.npy` |

## 3. 损失函数讲解口径

训练目标可以概括为：

```text
L = L_sup + lambda_replay L_replay + lambda_prior L_prior
    + lambda_sparse |alpha| + lambda_delta ||Delta W||
```

- `L_sup`：KITTI 有标签监督，来自多轮视差预测的 sequence loss。
- `L_replay`：冻结 FFS teacher 在无标签跨域样本上产生伪标签，用 photometric、augmentation、edge-risk 等可靠性项加权，减少目标域适配导致的遗忘。
- `L_prior`：在 FFS base 高置信区域保持学生输出接近原始 FFS，避免把基础几何先验完全破坏。
- `|alpha|`：让 RepGeo 残差贡献系数稀疏，抑制无效残差。
- `||Delta W||`：约束折叠残差规模，保持模型在原权重附近校准，而不是完全漂移。

## 4. RepGeo 单元怎么讲

训练时，卷积层被替换为：

```text
y = Conv(x, W0) + alpha * scaling * Conv_up(Conv_down(x))
```

其中 `Conv_down/Conv_up` 是低秩残差，`alpha` 是该残差单元的可学习贡献系数。部署时将残差严格折叠：

```text
W* = W0 + alpha * scaling * DeltaW
```

折叠后模型只保留普通 Conv2d/Conv3d 权重，不保留 LoRA wrapper、router 或输入相关 if-else。因此它不是推理阶段外挂模型，而是一个静态 FFS-style checkpoint。

## 5. 命令说明

所有命令都在工程根目录执行。PowerShell 示例：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\00_check_demo_assets.ps1
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\01_inference_no_window.ps1
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\02_eval_only_smoke.ps1
```

如果现场没有完整 KITTI 数据，只跑 `00_check_demo_assets.ps1` 和 `01_inference_no_window.ps1` 即可；训练与评估命令需要 `data_scene_flow/training` 存在。

## 6. 现场答辩注意

- 不说“超过所有 SOTA”；说“在 Fast-FoundationStereo 这个实时 backbone 上做目标域校准”。
- 不说“所有数据集都提升”；说“KITTI 强适配提升明显，跨域用保守 gamma 或 RGF 边界解释”。
- 不说“只是 LoRA”；说“训练期可学习几何残差，部署期静态折叠为 FFS-style checkpoint”。
- 老师如果问能否复现，直接展示 `commands` 脚本和 `source_snapshot` 代码索引。

