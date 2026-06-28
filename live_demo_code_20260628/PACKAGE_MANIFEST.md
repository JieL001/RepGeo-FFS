# 代码包清单

## 包含内容

| 路径 | 内容 |
| --- | --- |
| `00_快速上手指南.md` | 现场运行步骤 |
| `README_现场演示代码.md` | 代码包总说明 |
| `CODE_WALKTHROUGH_现场讲解.md` | 面向答辩问答的代码讲解稿 |
| `WEIGHTS_AND_RUNS_权重与演示路线.md` | 原文权重和改进权重的演示路线 |
| `TRAINING_GUIDE_训练启动与代码细节.md` | 深度学习课程用训练启动、损失函数和日志说明 |
| `TEACHER_CHECKLIST_验收问答准备.md` | 按老师现场验收问题准备的问答清单 |
| `commands/` | 可直接运行的 PowerShell 命令和无窗口推理脚本 |
| `source_snapshot/` | 训练、模型、损失、推理、折叠相关源码快照 |
| `outputs/infer_base/` | 推理样例输出 |
| `outputs/infer_repgeo_target_gamma1/` | 我们的 RepGeo-Target 推理样例输出 |
| `outputs/infer_repgeo_calibrated_gamma0p01/` | 我们的 RepGeo-Calibrated 推理样例输出 |
| `outputs/eval_smoke.json` | 小样本评估样例结果 |
| `outputs/eval_smoke_base.json` | 原文 FFS 小样本评估样例 |
| `outputs/eval_smoke_repgeo_target.json` | RepGeo-Target 小样本评估样例 |
| `outputs/repgeo_merged_demo_report.json` | 静态折叠样例报告 |

## 未放入压缩包的大文件

以下文件是运行生成的临时权重，体积较大，不放入干净版压缩包：

| 类型 | 说明 |
| --- | --- |
| smoke 训练 checkpoint | 可由 `03_train_smoke_repgeo.ps1` 重新生成 |
| merge demo checkpoint | 可由 `04_merge_repgeo_static.ps1` 重新生成 |
| Python 缓存 | 可自动生成，无需保存 |

## 运行依赖

代码包依赖完整工程中的以下资源：

```text
weights/23-36-37/model_best_bp2_serialize.pth
demo_data/left.png
demo_data/right.png
data_scene_flow/training/...
```

如果只做推理演示，不需要 KITTI 训练数据；如果要跑 eval 或 train smoke，需要 `data_scene_flow/training` 存在。

