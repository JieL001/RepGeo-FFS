# 训练启动与代码细节说明

本文件专门用于深度学习课程现场说明：模型如何训练、训练命令怎么启动、损失函数在哪里实现、训练日志应该怎么看。

## 1. 训练前准备

进入工程根目录并激活环境：

```powershell
cd RepGeo-FFS
conda activate ffs
```

检查环境和数据：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\00_check_demo_assets.ps1
```

训练需要以下数据目录存在：

```text
data_scene_flow/training/image_2
data_scene_flow/training/image_3
data_scene_flow/training/disp_occ_0
data_scene_flow/training/disp_noc_0
```

这些对应 KITTI stereo 训练集的左图、右图、遮挡视差标注和非遮挡视差标注。

## 2. 现场训练演示命令

现场建议运行短训练命令：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\03_train_smoke_repgeo.ps1
```

这个命令只使用：

```text
4 个训练样本
2 个验证样本
1 个 epoch
256x768 crop
valid_iters=2
```

作用不是重新得到最终指标，而是证明以下训练路径真实可执行：

```text
KITTI 数据读取
-> FFS 前向传播
-> RepGeo/LoRA 残差注入
-> supervised loss
-> geometry prior loss
-> alpha/delta 正则
-> backward
-> optimizer step
-> validation
-> checkpoint 保存
```

## 3. 完整训练参考命令

完整训练命令较慢，不建议现场直接跑。可以先打印完整命令：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\09_train_full_reference.ps1
```

如需正式启动完整训练，增加 `-Run`：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\09_train_full_reference.ps1 -Run
```

完整训练默认设置：

| 参数 | 设置 | 含义 |
| --- | --- | --- |
| `--epochs` | `8` | 较完整的适配训练轮数 |
| `--crop_h / --crop_w` | `320 / 960` | KITTI 训练 crop 尺寸 |
| `--valid_iters` | `4` | FFS 迭代更新次数 |
| `--lora_adapt` | 开启 | 插入可训练低秩残差 |
| `--lora_targets` | `cost update upsample` | 在代价体、迭代更新、上采样相关模块插入残差 |
| `--preserve_loss_weight` | `0.05` | 保持冻结 FFS 几何先验 |
| `--repgeo_sparse_weight` | `1e-5` | alpha 稀疏正则 |
| `--repgeo_delta_weight` | `1e-4` | 残差权重幅度约束 |

## 4. 训练代码入口

训练入口在：

```text
scripts/train_kitti.py
```

现场可以按下面顺序打开源码：

| 代码位置 | 作用 |
| --- | --- |
| `parse_args()` | 定义训练参数 |
| `create_dataloaders()` | 构建 KITTI train / val dataloader |
| `configure_lora_adapters()` | 给 FFS 指定模块插入 RepGeo/LoRA 残差 |
| `train_one_epoch()` | 一轮训练主循环 |
| `validate()` | 计算 EPE、D1、bad3 等指标 |
| `save_checkpoint()` | 保存训练态 checkpoint |

训练主链路可以概括为：

```text
main()
  -> create_dataloaders()
  -> load_model_for_training()
  -> configure_lora_adapters()
  -> train_one_epoch()
      -> model(left, right)
      -> sequence_loss()
      -> visibility_aware_loss()
      -> compute_preserve_loss()
      -> repgeo_regularization()
      -> loss.backward()
      -> optimizer.step()
  -> validate()
  -> save_checkpoint()
```

## 5. 损失函数组成

本项目训练不是只加一个 LoRA。训练目标由多项组成：

```text
L_total =
  L_sequence
  + lambda_boundary L_boundary
  + lambda_prior L_prior
  + lambda_sparse |alpha|
  + lambda_delta ||DeltaW||
```

如果开启 replay 分支，还会加入：

```text
+ lambda_replay L_replay
```

### 5.1 `L_sequence`

位置：

```text
scripts/train_kitti.py::sequence_loss
```

作用：对 FFS 多次迭代输出的视差进行监督。后面迭代的预测权重更高，因为最终视差主要来自最后几轮 refinement。

### 5.2 `L_boundary`

位置：

```text
scripts/train_kitti.py::visibility_aware_loss
```

作用：边界、遮挡和可见区域的误差性质不同。该项用于增强边界区域和可见区域的训练约束。

### 5.3 `L_prior`

位置：

```text
scripts/train_kitti.py::compute_preserve_loss
```

作用：冻结原始 FFS 作为几何先验 teacher，让学生模型在高置信区域不要过度偏离原模型。这样可以缓解只适配 KITTI 后造成的跨域退化。

### 5.4 `|alpha|` 与 `||DeltaW||`

位置：

```text
scripts/train_kitti.py::repgeo_regularization
core/lora.py::repgeo_alpha_l1
```

作用：

```text
alpha 控制每个 RepGeo 残差单元的贡献强度。
|alpha| 约束用于减少无效残差。
||DeltaW|| 约束用于避免残差过大，保持模型在 FFS 原权重附近做校准。
```

## 6. RepGeo 残差如何参与训练

RepGeo/LoRA 残差单元在：

```text
core/lora.py
```

训练阶段：

```text
y = Conv(x, W0) + alpha * scaling * Conv_up(Conv_down(x))
```

其中：

| 符号 | 含义 |
| --- | --- |
| `W0` | FFS 原始卷积权重 |
| `Conv_down / Conv_up` | 低秩残差参数 |
| `alpha` | 可学习贡献系数 |
| `scaling` | LoRA scaling，通常为 `alpha_lora / rank` |

部署阶段会折叠成：

```text
W* = W0 + alpha * scaling * DeltaW
```

折叠代码在：

```text
scripts/merge_lora_for_inference.py
```

核心检查输出：

```text
before merge: lora_modules=72
after merge: lora_modules=0
```

这说明最终模型没有 LoRA wrapper、没有 router、没有额外推理分支。

## 7. 训练日志怎么看

运行 `03_train_smoke_repgeo.ps1` 后，会看到类似输出：

```text
LoRA: rank=4 alpha=8.0 dropout=0.05 targets=['cost', 'update', 'upsample'] adapters=72
trainable params: 0.26M / 17.93M
epoch 00 step 0000/0004 loss=...
vis=...
bnd=...
preserve=...
alpha_l1=...
delta_l2=...
epoch 00 done: train_loss=... val_EPE=... val_D1=... val_bad3=...
```

关键字段解释：

| 字段 | 含义 |
| --- | --- |
| `adapters=72` | 插入了 72 个残差单元 |
| `trainable params=0.26M / 17.93M` | 只训练约 1.45% 参数 |
| `vis` | 可见区域监督损失 |
| `bnd` | 边界区域损失 |
| `preserve` | 几何先验保持损失 |
| `alpha_l1` | RepGeo alpha 稀疏项 |
| `delta_l2` | 残差权重幅度项 |
| `val_EPE` | 验证集平均端点误差 |
| `val_D1 / val_bad3` | KITTI bad-pixel 类指标 |

## 8. 训练态 checkpoint 与部署态 checkpoint 区别

| 类型 | 说明 | 典型路径 |
| --- | --- | --- |
| 训练态 checkpoint | 保留 LoRA/RepGeo wrapper，可继续训练 | `repgeo_student/model_best_epe.pth` |
| 部署态 checkpoint | 已折叠为普通 FFS 权重，可直接推理 | `repgeo_gamma_1.pth` |

现场重点展示部署态 checkpoint，因为它更符合最终模型：

```text
输入左右图 -> 纯 FFS-style forward -> 输出视差
```

## 9. 现场建议讲法

> 训练时，我们冻结大部分 FFS 主干，只在 cost、update 和 upsample 相关卷积上插入可学习残差。损失函数由 KITTI 监督、边界约束、原始 FFS 几何先验保持和 RepGeo 残差正则组成。训练结束后，残差通过 `merge_lora_for_inference.py` 折叠回 Conv2d/Conv3d 权重，因此最终推理时不需要 LoRA 分支或 router。


