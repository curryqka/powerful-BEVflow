# 上游仓库

## 克隆信息

- 仓库：`https://github.com/merlresearch/MotionNet`，位置 `third_party/MotionNet`，只读。
- 固定 commit：`aea30a1777f06e83f65fa4cb3e93ae95519d9a5b`（"Create release 1.0.0"）。
- License：AGPL-3.0-or-later（MERL），部分文档 CC-BY-NC-SA-4.0。学术用途。
- 作者页 `pxiangwu/MotionNet` 经 GitHub API 核实只有 `readme.md` 与 `teaser`，无代码。以 MERL 版为准。

## 上游脚本对照

| 事项 | 上游脚本 |
|---|---|
| 生成数据 | `data/gen_data.py` |
| 训练（STPN 主模型，加权损失） | `train_multi_seq.py` |
| 训练（单序列） | `train_single_seq.py` |
| 评测位移与分类 | `eval.py` |
| 可视化 | `plots.py` |
| MGDA 变体 | `min_norm_solvers.py`，`train_*_MGDA.py` |

MGDA 相关文件按上游 README 说明只能在 PyTorch 1.1 上运行。本项目不用，不迁移，不打补丁。

## 上游 README 的环境要求

- Python 3，pyquaternion，Matplotlib，PIL，numpy，cv2，tqdm。
- PyTorch >= 1.1（MGDA 部分锁定 1.1）。实际按 GPU sm_89 装新 torch，见 `env.md`。

## 官方权重

- 来源：Zenodo 记录 8215963（https://doi.org/10.5281/zenodo.8215963）。
- 文件：`MotionNet.zip`，162232742 字节。
- md5：`eb3fbb88a8ab2fcbeea0414d5a6125ef`（2026-10-16 下载后校验一致）。
- 位置：`$EXP/_models/motionnet_official/`，解压内容：`model.pth`（95.4MB，主模型权重，本项目用），`model_MGDA.pth`（MGDA 变体，不用），`README.md`。

## 预期需要补丁的地方

- `np.int` / `np.float` 等 NumPy 已移除的别名。
- `torch.load` 的 `weights_only` 默认值变化。
- 过时的 DataLoader 参数。
- 路径硬编码。
- cv2：装的是 `opencv-python-headless`（服务器无 GUI），import 兼容。
