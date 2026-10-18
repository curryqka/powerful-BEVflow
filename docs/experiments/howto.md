# 如何复现 R1 全流程

一条命令：

```bash
RUN_ID=<日期>_R1B_motionnet_s0 bash scripts/reproduce_r1.sh 4
```

五个阶段：数据生产（8 分片并行 ~1.2h，已有 data_card 则跳过）→ 训练（~14h，--resume auto 断点安全）→ last-10 评测 → test 终评（bs1）→ 可视化（4 个固定样本）。

前提（一次性，见 docs/setup/）：

- uv 环境：`docs/setup/env.md` 的重建命令；每个 shell 先 `export UV_CACHE_DIR=$EXP/_cache/uv`。
- 原始 nuScenes 在 `$RAW_NUSCENES`（总控提示词第 0 节）。
- PYTHONPATH 由各脚本内部设置（vendored devkit），无需手动 export。

单步手动执行见各脚本 `--help`：`gen_data.py`（支持 `--scenes` 子集），`train.py`（`--set` 点号覆盖），`eval.py`（bs1 为逐样本严格口径），`eval_per_ckpt.py`，`viz.py`。注册表随训练/评测自动写入 `$RESULTS/registry.jsonl`。

R1A 对照（上游 README 官方配方，bs8 一致性全开）：

```bash
RUN_ID=<日期>_R1A_motionnet_s1 CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/exp/R1A_official.yaml --set run.run_id=$RUN_ID train.seed=1
```

注意：该配方存在种子依赖的零解塌缩风险（见 docs/reports/R1.md 第 5 节），塌缩信号为 epoch 1 内一致性损失降至机器零且 disp 分项停滞。
