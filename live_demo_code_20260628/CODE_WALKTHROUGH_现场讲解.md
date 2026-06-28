# 现场代码讲解稿

用途：老师现场要求查看“训练、损失函数、推理、模型改进在哪里实现”时，按本文件顺序打开代码即可。建议先激活可用 CUDA 的 `ffs` conda 环境，再执行命令脚本。

```powershell
conda activate ffs
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\00_check_demo_assets.ps1
```

如果检查脚本显示 `cuda_available=False`，说明当前 `python` 不是 FFS 环境，不要用它跑推理或训练。

## 1. 网络前向：Fast-FoundationStereo 主干

打开：

```text
source_snapshot/core__foundation_stereo.py
```

重点看：

| 位置 | 作用 |
| --- | --- |
| `FastFoundationStereo` 类，约第 135 行 | FFS 主模型定义 |
| `self.corr_stem`，约第 164 行 | 代价体预处理 |
| `self.cost_agg`，约第 171 行 | 多尺度代价聚合 |
| `self.update_block`，约第 146 行 | 迭代更新视差 |
| `self.spx_2_gru / self.spx_gru`，约第 159 行 | 上采样与边缘细化 |
| `forward(...)`，约第 210 行 | 双目输入到视差输出的完整流程 |

现场讲法：

> 原 FFS 的推理结构没有被重写。输入左右图后，模型先提取特征并构建代价体，然后通过代价聚合和迭代更新得到视差，最后用上采样模块输出高分辨率视差图。我们的改进主要插入在 cost、update、upsample 这几类卷积层附近，不改变主干的数据流。

## 2. RepGeo/LoRA 单元：残差从哪里来

打开：

```text
source_snapshot/core__lora.py
```

重点看：

| 位置 | 作用 |
| --- | --- |
| `LoRAConv2d`，约第 68 行 | 2D 卷积残差包装 |
| `LoRAConv3d`，约第 118 行 | 3D 卷积残差包装 |
| `repgeo_alpha`，约第 86 / 132 行 | 每个残差单元的可学习贡献系数 |
| `forward`，约第 113 / 159 行 | `base(x) + alpha * low-rank residual(x)` |
| `repgeo_alpha_l1`，约第 234 行 | alpha 稀疏正则 |
| `repgeo_alpha_stats`，约第 241 行 | 统计各单元贡献 |

核心代码逻辑：

```text
output = base_conv(x)
       + lora_up(lora_down(dropout(x))) * scaling * repgeo_alpha
```

现场讲法：

> 这部分最开始形式上类似 Conv-LoRA，但 RepGeo 的重点不是保留 LoRA 分支，而是把它作为训练期几何残差单元。每个单元有可学习的 alpha，用来控制这条残差是否有贡献。训练结束后，残差会被折叠回原始卷积权重。

## 3. 训练入口：完整训练路径

打开：

```text
source_snapshot/scripts__train_kitti.py
```

重点看：

| 位置 | 作用 |
| --- | --- |
| `parse_args`，约第 56 行 | 所有训练参数入口 |
| `configure_lora_adapters`，约第 254 行 | 指定在哪些模块插入残差单元 |
| `train_one_epoch`，约第 886 行 | 前向、损失计算、反传、优化器 step |
| `validate`，约第 1130 行 | EPE / D1 / bad3 等指标验证 |
| `main`，约第 1255 行 | 加载数据、加载权重、启动训练或评估 |

现场讲法：

> 训练不是单独脚本拼出来的，入口统一在 `train_kitti.py`。它可以做 eval-only，也可以做 RepGeo/LoRA 小样本训练。现场 smoke run 只跑少量样本，证明数据读取、模型前向、损失、反传路径都可执行；完整结果来自更长训练。

## 4. 损失函数：不是只加 LoRA

打开：

```text
source_snapshot/scripts__train_kitti.py
```

重点看：

| 函数 | 位置 | 含义 |
| --- | ---: | --- |
| `sequence_loss` | 约第 272 行 | 多轮视差预测监督损失 |
| `visibility_aware_loss` | 约第 329 行 | 可见区域与边界区域加权 |
| `compute_pseudo_self_distill_loss` | 约第 480 行 | reliability replay，自蒸馏保持 |
| `compute_preserve_loss` | 约第 560 行 | 冻结 FFS 几何先验保持 |
| `repgeo_regularization` | 约第 797 行 | alpha 稀疏与残差规模约束 |

可写在白板上的总目标：

```text
L_total =
  L_supervised
  + lambda_replay * L_replay
  + lambda_prior * L_prior
  + lambda_sparse * |alpha|
  + lambda_delta * ||DeltaW||
```

### 4.1 KITTI 监督损失

`sequence_loss` 对每轮预测视差都计算误差，后期预测权重更高。它对应目标域有标签训练，是 KITTI 提升的主要来源。

### 4.2 Reliability replay

`compute_pseudo_self_distill_loss` 使用冻结 FFS teacher 在跨域无标签样本上生成伪标签，并结合可靠性权重。可靠性来源包括：

- augmentation consistency：增强前后预测是否一致；
- photometric consistency：右图重投影回左图后是否匹配；
- edge risk：边缘/遮挡附近降低伪标签权重。

讲法：

> 这不是在训练 KITTI 时直接学习其他数据集标签，而是用冻结 FFS 的高置信预测作为跨域几何先验，防止目标域适配把原来的泛化能力完全破坏。

### 4.3 几何先验保持

`compute_preserve_loss` 会比较学生模型与冻结 FFS base 的输出，尤其关注高置信区域。作用是把更新限制在 FFS 原来的几何先验附近，而不是任意漂移。

讲法：

> 强适配会提高 KITTI，但也可能破坏 ETH3D/Middlebury。prior loss 的作用是约束学生模型不要偏离原 FFS 的可靠几何判断。

### 4.4 RepGeo 正则

`repgeo_regularization` 会统计 `repgeo_alpha` 和残差规模：

- alpha 稀疏：减少无效残差；
- delta 约束：避免残差太大；
- 贡献审计：可以统计哪些 residual unit 被保留。

## 5. 静态折叠：最终模型为什么没有额外分支

打开：

```text
source_snapshot/scripts__merge_lora_for_inference.py
```

重点看：

| 函数 | 位置 | 作用 |
| --- | ---: | --- |
| `_merge_conv2d` | 约第 129 行 | 将 2D 低秩残差折叠成普通 Conv2d 权重 |
| `_merge_conv3d` | 约第 151 行 | 将 3D 低秩残差折叠成普通 Conv3d 权重 |
| `merge_lora_modules` | 约第 173 行 | 遍历模型并替换所有 wrapper |

核心实现：

```text
DeltaW = B @ A * scaling * alpha
W* = W0 + DeltaW
```

对应代码：

- Conv2d delta：约第 143 行；
- Conv3d delta：约第 165 行；
- 写回 merged weight：约第 144 / 166 行。

现场讲法：

> RepGeo 的最终部署模型不是带 LoRA wrapper 的模型。训练结束后，低秩残差被乘上 alpha 并加回原始卷积核，得到新的普通卷积权重。导出的 checkpoint 仍然走 FFS 原始 forward，没有 router，没有动态分支，没有输入相关 if-else。

## 6. 推理脚本：现场不会弹窗卡住

打开：

```text
commands/infer_no_window.py
```

运行：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\01_inference_no_window.ps1
```

输出：

```text
live_demo_code_20260628/outputs/infer_base/left.png
live_demo_code_20260628/outputs/infer_base/right.png
live_demo_code_20260628/outputs/infer_base/disp.npy
live_demo_code_20260628/outputs/infer_base/disp_vis.png
live_demo_code_20260628/outputs/infer_base/infer_board.png
```

讲法：

> 原始 demo 脚本会弹出 OpenCV 窗口，现场容易卡住。这里单独准备了无窗口推理脚本，保存左右图、视差数组和可视化拼图，方便现场展示。

## 7. 现场命令选择

### 必跑：资源检查

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\00_check_demo_assets.ps1
```

### 必跑：推理

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\01_inference_no_window.ps1
```

### 可选：小评估

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\02_eval_only_smoke.ps1
```

### 可选：小训练

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\03_train_smoke_repgeo.ps1
```

### 可选：折叠导出

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\04_merge_repgeo_static.ps1
```

### 可选：测速

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\05_runtime_3090_style.ps1
```

## 8. 现场问答口径

| 老师问题 | 回答 |
| --- | --- |
| 你们训练代码在哪里？ | `scripts/train_kitti.py`，入口是 `main`，训练循环是 `train_one_epoch`。 |
| 损失函数在哪里？ | 同一个文件里，主要是 `sequence_loss`、`compute_pseudo_self_distill_loss`、`compute_preserve_loss`、`repgeo_regularization`。 |
| RepGeo 是不是 LoRA？ | 形式上用低秩残差实现，但最终目标不是保留 LoRA 分支，而是训练期校准、部署期折叠成静态 Conv 权重。 |
| 折叠到了哪里？ | 折叠到原 FFS 的 Conv2d/Conv3d 权重，主要是 cost、update、upsample 相关模块。 |
| 推理有没有 router？ | RepGeo compiled checkpoint 没有 router，也没有 LoRA wrapper，前向仍是普通 FFS。 |
| 为什么不现场跑完整训练？ | 完整训练需要较长 GPU 时间，现场用 smoke run 展示代码路径，完整训练日志和结果表在报告中。 |


