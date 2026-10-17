# 冒烟记录

冒烟脚本：`scripts/smoke.sh`，两种模式：`SMOKE_MODE=upstream`（上游脚本，重构前基线）与 `SMOKE_MODE=new`（bevflow 新入口，重构后），检查项完全相同。目录名带模式前缀。

## 通过记录

- 2026-10-17 00:23，upstream 模式全项 PASS：`$RESULTS/_smoke/20261017_002303/`。
- 2026-10-17 02:09 起，new 模式记录见下方（重构后入口：scripts/train.py，scripts/eval.py，scripts/viz.py，bevflow losses）。
- 数据：mini 划分，seed 0 从 train split 抽 10 个场景（6/2/2），脚本 `scripts/make_mini_split.py`。本机无 nuScenes mini 原始数据，用 trainval 子集代替（决定记录在 `recon.md`）。

## 阈值调整（初始设定的修改）

- S4 的 cls 指标允许 nan：mini val 只有 2 个场景，Bike 类整类无格子，混淆矩阵该行除零。metrics.json 记 null。全量 val/test 评测中 nan 不允许，出现即查。
- 其余阈值（S3 1e-6，S2b 损失减半，S5 尺寸与 5KB）沿用初始设定，未动。

## 上游兼容问题（本轮发现，均已解决）

| 问题 | 处理 |
|---|---|
| `check_folder` 用非递归 `os.mkdir`，savepath 父目录要预先存在 | 调用方先 `mkdir -p`，未打补丁 |
| `np.bool`/`np.int` 共 18 处 | 补丁 `0001-numpy-removed-aliases.patch` |
| 损失处 `.view` 遇 torch 2.x 非连续张量报错 | 补丁 `0002-loss-disp-view-to-reshape.patch` |
| train 的 `--logpath` 会在其下再建 `train_multi_seq/<时间戳>/`，ckpt 与 log.txt 在里面 | smoke.sh 已按此找产物 |
| `gen_data.py` 按 CWD 相对路径读 `data/split.npy` | 用独立 scratch CWD（`$EXP/_cache/smoke_cwd`）放 mini split，不动上游 |
| eval 加载器 `DatasetSingleSeq` 只认扁平 npy 目录，gen_data 输出是子目录 | `scripts/flatten_eval_data.py` 拍平，P4 全量沿用 |
| `scheduler.step()` 在 `optimizer.step()` 之前 | 上游固有行为，warning 无害，不改 |

## 评测协议（motionnet_v1）定义

- 速度三档：static（每帧位移 ≤ 0.01 m，即 0.2 m/s），slow（0–5 m/s），fast（5–20 m/s），与 `eval.py` 一致。
- EPE 的 mean/median：取 5 个评测帧（0.2–1.0s，间隔 0.2s）上误差均值的平均、中位数均值的平均。逐帧数组也存进 metrics.json，改口径不用重跑。
- OA = 像素分类准确率；MCA = 类别准确率的均值（上游称 mean instance acc）。
- 冒烟 ckpt 只训了 104 迭代，EPE 0.5/1.4/3.9 m 与 OA 0.06 只证明管线连通，不是结果。

## 冒烟可视化

4 个固定样本（seed 0 抽样）：`$RESULTS/_smoke/20261017_002303/viz/`，每个样本 4 张图（GT，预测，误差，静态格子上的预测幅值）。色轮与色标固定，HSV 编码。
