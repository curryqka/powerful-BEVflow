# 上游补丁记录

补丁文件在 `patches/`，应用到 `third_party/MotionNet`（独立 git 克隆，不入库）。重新克隆后按序 `git apply`。

| 补丁 | 改了什么 | 为什么 |
|---|---|---|
| `0001-numpy-removed-aliases.patch` | `np.bool` → `bool`，`np.int` → `int`，4 个文件（nuscenes_dataloader，gen_data，data_utils，plots）；eval.py 的同类修改并入 0003 | 这些别名 numpy 1.24 起移除，本机 numpy 2.2.6 直接报错。纯别名替换，行为不变 |
| `0002-loss-disp-view-to-reshape.patch` | `train_multi_seq.py` 损失处 `.view(...)` → `.reshape(...)`，1 处 | torch 2.x 的 TensorIterator 保留非连续输入的布局，SmoothL1 输出非连续，`.view` 报错。torch 1.1 输出总是连续。reshape 数值等价 |
| `0003-eval-confusion-matrix-labels.patch` | `eval.py` 混淆矩阵加 `labels=list(range(1, 6))`，1 处 | 小样本评测时某类别可能整类不出现，sklearn 推断标签数不足，上游硬编码 5 类索引越界。注意 GT/Pred 值域是 1–5（argmax+1），标签必须给 1..5；首版误给 0..4 导致逐类准确率整体错位（官方权重评测中发现并已修正），全量数据上与上游推断行为一致 |

尚未遇到但预期可能需要的：`torch.load` 的 `weights_only` 默认值变化（torch 2.6+ 加载旧 ckpt 时）。
