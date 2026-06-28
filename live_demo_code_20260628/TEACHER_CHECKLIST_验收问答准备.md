# 老师验收问答准备

本文件按“老师现场可能怎么验收”组织。每个问题都给出：应该展示什么、运行什么命令、怎么回答。

## 1. 最可能的现场验收流程

如果我是老师，通常会按这个顺序检查：

1. 你们是否真的能跑原论文代码？
2. 你们是否真的改了训练代码，而不是只改了报告？
3. 你们的损失函数在哪里实现？
4. 你们的改进模型权重在哪里，和原文权重怎么区分？
5. 你们的改进是否真的参与推理？
6. 你们的训练过程是否可复现？
7. 你们的结果是否有数据集、指标和日志支撑？
8. 你们的最终模型有没有额外推理分支？
9. 你们知道这个方法的边界在哪里吗？

现场不要先讲很多概念，先用代码和命令证明。

## 2. 5 分钟验收演示顺序

### 第一步：检查环境和资源

```powershell
conda activate ffs
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\00_check_demo_assets.ps1
```

要让老师看到：

```text
cuda_available=True
weights/23-36-37/model_best_bp2_serialize.pth OK
repgeo_gamma_1.pth OK
data_scene_flow/training/... OK
```

回答要点：

> 这里检查的是原文 FFS 权重、我们的 RepGeo 权重、KITTI 数据和核心脚本。通过后说明现场代码不是孤立文档，而是能读取真实工程资源。

### 第二步：跑原文 FFS 推理

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\01_inference_no_window.ps1
```

展示：

```text
live_demo_code_20260628\outputs\infer_base\infer_board.png
```

回答要点：

> 这一步加载的是原文发布的 FFS base 权重，证明原模型从左右图输入到视差输出可以跑通。

### 第三步：跑我们的 RepGeo 推理

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\06_inference_repgeo_target.ps1
```

展示：

```text
live_demo_code_20260628\outputs\infer_repgeo_target_gamma1\infer_board.png
```

回答要点：

> 这一步加载的是我们折叠后的 RepGeo-Target 静态权重，不是原文权重。推理代码和原 FFS 一样，说明最终模型是静态 FFS-style checkpoint。

### 第四步：跑训练 smoke

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\03_train_smoke_repgeo.ps1
```

要让老师看到日志：

```text
LoRA: rank=4 alpha=8.0 targets=['cost','update','upsample'] adapters=72
trainable params: 0.26M / 17.93M
loss=...
vis=...
bnd=...
preserve=...
alpha_l1=...
delta_l2=...
val_EPE=...
val_D1=...
```

回答要点：

> 这一步只跑小样本，不是为了复现最终数字，而是证明训练代码、损失函数、反向传播和验证指标路径都能执行。完整训练结果来自长时间服务器实验。

### 第五步：展示静态折叠

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\04_merge_repgeo_static.ps1
```

要让老师看到：

```text
before merge: lora_modules=72
after merge: lora_modules=0
```

回答要点：

> 训练阶段有 RepGeo/LoRA 残差单元，部署阶段会折叠回普通 Conv2d/Conv3d 权重。最终 checkpoint 没有 LoRA wrapper、没有 router、没有输入相关 if-else。

## 3. 高频问题与标准回答

### Q1：你们复现的是哪篇论文？

回答：

> 复现主线是 Fast-FoundationStereo。它是实时零样本双目立体匹配模型，目标是在保持 FoundationStereo 泛化能力的同时提高推理速度。我们先跑通原文权重、数据读取、推理和 KITTI/ETH3D/Middlebury 评估，再在它上面做目标域校准改进。

展示：

```text
source_snapshot/core__foundation_stereo.py
commands/01_inference_no_window.ps1
```

### Q2：原文模型代码从哪里开始？

回答：

> 主模型在 `core/foundation_stereo.py` 的 `FastFoundationStereo`。输入左右图后，经过特征提取、代价体构建、代价聚合、迭代更新和上采样，最后输出视差。

展示：

```text
source_snapshot/core__foundation_stereo.py
```

重点指：

```text
FastFoundationStereo.forward
corr_stem / cost_agg
update_block
spx_2_gru / spx_gru
```

### Q3：你们到底改了模型哪里？

回答：

> 我们没有重写 FFS 主干，而是在 cost、update、upsample 相关卷积层上加入训练期可学习残差单元。形式上是低秩残差，但最终会折叠回原卷积，所以部署时仍是 FFS-style 静态模型。

展示：

```text
source_snapshot/core__lora.py
source_snapshot/scripts__merge_lora_for_inference.py
```

重点指：

```text
LoRAConv2d / LoRAConv3d
repgeo_alpha
merge_lora_modules
```

### Q4：这不就是 LoRA 吗？

回答：

> 基础实现借用了低秩残差形式，但目标不是保留 LoRA 分支。普通 LoRA 是推理时仍可作为 adapter 使用；RepGeo 的重点是训练期残差校准、部署期静态折叠。我们还加入了几何先验保持、alpha 稀疏和残差规模约束，所以它在训练目标和部署形态上都不只是普通 LoRA。

可以写在纸上：

```text
训练：y = Conv(x,W0) + alpha * scaling * Conv_up(Conv_down(x))
部署：W* = W0 + alpha * scaling * DeltaW
```

### Q5：损失函数在哪里？

回答：

> 训练损失集中在 `scripts/train_kitti.py`。核心包括 KITTI supervised sequence loss、边界/可见性损失、冻结 FFS 几何先验保持损失、RepGeo alpha 稀疏和残差幅度约束。

展示：

```text
source_snapshot/scripts__train_kitti.py
```

重点指：

```text
sequence_loss
visibility_aware_loss
compute_preserve_loss
repgeo_regularization
train_one_epoch
```

### Q6：怎么启动训练？

回答：

> 现场用 smoke training 展示完整训练路径，完整训练命令通过 `09_train_full_reference.ps1` 打印，避免误启动长训练。

运行：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\03_train_smoke_repgeo.ps1
```

展示完整命令：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\09_train_full_reference.ps1
```

### Q7：完整训练和现场训练有什么区别？

回答：

> 现场训练只用 4 个样本、1 个 epoch，用于证明代码路径。完整训练使用更大 crop、更多 epoch、完整数据划分和服务器 GPU，最终结果以实验报告和保存日志为准。

不要说现场 smoke 结果就是最终结果。

### Q8：你们的改进权重在哪里？

回答：

> 原文权重是 `weights/23-36-37/model_best_bp2_serialize.pth`。我们的最终 RepGeo 静态权重在 `output_eval/stage29_repgeo_gamma_sweep_20260617/`，其中 `repgeo_gamma_1.pth` 是目标域强适配版，`repgeo_gamma_0p01.pth` 是保守校准版。

展示：

```text
WEIGHTS_AND_RUNS_权重与演示路线.md
```

### Q9：你们的最终模型推理时有没有 router？

回答：

> RepGeo compiled checkpoint 没有 router。训练阶段可以有残差单元，部署阶段折叠后只剩普通 FFS-style forward。

证据：

```text
04_merge_repgeo_static.ps1 输出 after merge: lora_modules=0
```

### Q10：为什么还保留 FFSOmega-R？

回答：

> FFSOmega-R 是中间适配器基线，用来验证 reliability replay 和 RGF 边界。最终答辩主线可以放在 RepGeo-FFS，因为它把训练期校准残差折叠成静态模型，更适合现场说明“改进模型可以直接部署”。

### Q11：你们有没有和原模型定量对比？

回答：

> 有。正式报告中使用 KITTI local-val、official private-preview，以及 ETH3D/Middlebury 跨域检查。现场为了节省时间，用 `08_eval_base_vs_repgeo_smoke.ps1` 在同一小样本上分别评估原 FFS 和 RepGeo，证明评估代码路径一致。

运行：

```powershell
powershell -ExecutionPolicy Bypass -File live_demo_code_20260628\commands\08_eval_base_vs_repgeo_smoke.ps1
```

### Q12：为什么不现场跑完整 KITTI？

回答：

> 完整 KITTI local-val 和 official submission 需要较长时间，不适合现场。现场只验证代码路径和权重加载；完整结果在报告中给出，并保留日志和提交包。

### Q13：你们有没有跨域退化？

回答：

> 有，强适配版本在 KITTI 上收益明显，但直接用于 ETH3D/Middlebury 会退化。因此我们报告中区分 RepGeo-Target 和 RepGeo-Calibrated，并保留 FFSOmega-RGF 作为有效性边界分析。这是方法边界，不应夸成所有数据集都更优。

### Q14：为什么深度学习课程也能用这个项目？

回答：

> 这个项目包含完整深度学习流程：网络结构、预训练权重、数据集、监督训练、参数高效适配、损失函数设计、反向传播、模型导出、推理和指标评估。不是只跑 demo。

## 4. 如果老师要求现场看训练代码

按这个顺序打开：

1. `source_snapshot/scripts__train_kitti.py`
2. 搜索 `def train_one_epoch`
3. 搜索 `loss.backward`
4. 搜索 `optimizer.step`
5. 搜索 `sequence_loss`
6. 搜索 `compute_preserve_loss`
7. 搜索 `repgeo_regularization`

回答：

> 这里能看到训练循环中先前向得到多轮视差预测，再计算监督损失、几何先验保持和 RepGeo 正则，最后执行反向传播和优化器更新。

## 5. 如果老师要求现场看模型改进代码

按这个顺序打开：

1. `source_snapshot/core__lora.py`
2. 搜索 `class LoRAConv2d`
3. 搜索 `repgeo_alpha`
4. 搜索 `def forward`
5. 打开 `source_snapshot/scripts__merge_lora_for_inference.py`
6. 搜索 `_merge_conv2d`
7. 搜索 `_merge_conv3d`

回答：

> 训练时每个残差单元有 `repgeo_alpha` 控制贡献，部署时通过 `_merge_conv2d/_merge_conv3d` 把低秩残差乘上 alpha 后加回原始卷积核。

## 6. 如果老师要求现场看结果文件

打开：

```text
live_demo_code_20260628/outputs/infer_base/infer_board.png
live_demo_code_20260628/outputs/infer_repgeo_target_gamma1/infer_board.png
live_demo_code_20260628/outputs/eval_smoke_base.json
live_demo_code_20260628/outputs/eval_smoke_repgeo_target.json
live_demo_code_20260628/outputs/repgeo_merged_demo_report.json
```

说明：

> 这些是现场实际跑出来的小样本输出，用于证明脚本可执行。正式论文/报告结果用完整数据集表格。

## 7. 不能这样回答

避免以下说法：

- “我们超过所有 SOTA。”
- “所有数据集都提升。”
- “这个 smoke 结果就是最终指标。”
- “RepGeo 完全不是 LoRA。”
- “router 是最终模型的一部分。”

更准确的说法：

- “我们是在实时 FFS backbone 上做目标域校准。”
- “KITTI 目标域提升明显，跨域需要保守校准或边界分析。”
- “RepGeo 使用低秩残差实现，但最终折叠为静态卷积权重。”
- “现场 smoke run 用于验证代码路径，完整结果来自完整实验。”


