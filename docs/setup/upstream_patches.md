# 上游补丁记录

补丁文件在 `patches/`，应用到 `third_party/MotionNet`（独立 git 克隆，不入库）。重新克隆后按序 `git apply`。

| 补丁 | 改了什么 | 为什么 |
|---|---|---|
| `0001-numpy-removed-aliases.patch` | `np.bool` → `bool`，`np.int` → `int`，共 18 处，5 个文件（nuscenes_dataloader，gen_data，data_utils，plots，eval） | 这些别名 numpy 1.24 起移除，本机 numpy 2.2.6 直接报错。纯别名替换，行为不变 |

尚未遇到但预期可能需要的：`torch.load` 的 `weights_only` 默认值变化（torch 2.6+ 加载旧 ckpt 时）。
