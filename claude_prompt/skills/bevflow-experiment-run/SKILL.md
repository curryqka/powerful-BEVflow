---
name: bevflow-experiment-run
description: 固定与 MotionNet 对齐的经验参数并跑第一轮实验时使用。覆盖参数卡，两套配置，MotionNet 官方权重下载与评测，训练运行，ckpt 保存，参考值对照，结果登记。
---

# 目标

得到第一轮可信的结果：参数有出处，数字来自日志，偏差有解释。

# 一、参数卡

从上游代码和 README 提取参数，写 `docs/reference/param_card.md`，每个值注明来源文件与行号。

| 组 | 内容 |
|---|---|
| 数据 | 体素大小，范围，高度 bin 数，过去帧数，未来帧数，自车补偿方式 |
| 模型 | 主干与通道，参数量，输出头 |
| 损失 | 各项权重，背景格子权重，平滑项 |
| 优化 | 优化器，学习率，调度，权重衰减，epoch，batch size |
| 评测 | 速度三档阈值，动静阈值，MCA 与 OA 的定义 |

项目文档里已有几个固定值，核对它们与上游是否一致，不一致就记下差异：

- 背景格子损失权重 0.005。
- 动静门控阈值约 0.2 m/s。
- 项目统一训练预算：45 个 epoch，单卡 bs4。24GB 放不下就用 bs2 加梯度累积 2。
- 主表 3 个种子。

# 二、两套配置

| 配置 | 用途 | 内容 |
|---|---|---|
| `exp/R1A_official.yaml` | 验证管线 | 上游默认训练配方 |
| `exp/R1B_budget.yaml` | 项目协议 | 数据与模型同 R1A，训练预算改为 45 epoch，bs4 或 bs2 加累积 2 |

R1B 之后是所有新方法共用的预算。两套配置只在训练预算上不同。

# 三、流程

1. **确认 MotionNet 官方权重。** P1 应已从 Zenodo 记录 8215963 下载并记录哈希。缺失时补下载：先打开页面确认文件名与大小，下载后记录哈希。网络不通就让我上传。
2. **评测官方权重。** 用我们的评测器在 test 上评测，这是整条管线的第一个对照点。
3. **训练 R1A。** 种子 0。评测最后 10 个 epoch 的 ckpt，写 `per_ckpt.json`。
4. **可视化。** 官方权重与 R1A 各出一套，样本固定，见 skill `bevflow-smoke-test`。
5. **训练 R1B。** 种子 0。
6. **补种子。** R1B 跑完后，自动提交种子 1 与种子 2。
7. **写报告。** `docs/reports/R1.md`。

# 四、训练运行

- 启动前 dry run 20 个迭代，确认损失，显存，日志，ckpt 路径。
- 用调度器模板 `scripts/slurm/train.sbatch`。没有调度器就用 tmux 加 nohup。
- 支持 `--resume auto`，从最新 ckpt 继续，被抢占后可恢复。
- 固定随机种子：python，numpy，torch，DataLoader worker。
- 日志格式见第七节。
- 每个 run 写 `run_manifest.json`，见 skill `bevflow-workspace-layout`。

# 五、ckpt 保存

项目里已知静态指标对 ckpt 很敏感，后面要做"最后 10 个 ckpt 的均值与标准差"。所以保存最后 10 个 epoch 的 ckpt，另存 `last.pt`。报告同时给最后一个 epoch 的数，以及最后 10 个的均值与标准差。要挑 ckpt，用验证集，并在配置里提前写好规则。

# 六、参考值对照

下表是 MotionNet 在三篇论文里的自报值，单位米，转引自项目文档，未逐篇核对原文。

| 来源 | static | slow | fast |
|---|---|---|---|
| WeakMotion 表 | 0.0201 | 0.2292 | 0.9454 |
| LEGO-Motion 表 | 0.0239 | 0.2467 | 1.0109 |
| SelfMotion 表 | 0.0256 | 0.2565 | 1.0744 |

三篇之间 static 的跨度约 27%，fast 约 14%，来自协议实现的差异。所以看包络，不看单个数：每档取三行的最小与最大。

官方权重评测值，以及 R1A 与 R1B 的结果，落在包络附近就算对得上，宽松一点没关系。对不上时，按这个顺序查：split，数据卡参数，评测阈值，自车补偿，背景权重，归一化。排查结论写进报告。这是对照，不是必须通过的门，对不上也继续推进后面的工作。

# 七、日志与结果

## 日志布局

```
$EXP/<exp_id>/<run_id>/
├── ckpt/
├── logs/
│   ├── train.jsonl         每 50 步一行，每个 epoch 末再一行
│   ├── stdout.log          训练进程的 stdout 与 stderr
│   ├── sched-<jobid>.out   调度器日志，没有调度器就没有这项
│   └── tb/                 tensorboard
├── config_snapshot.yaml
├── run_manifest.json
└── code.diff               仅当工作区有未提交改动时
```

`train.jsonl` 是日志的真源，tensorboard 只用来看曲线。两种行用 `type` 区分：

```json
{"type": "step", "epoch": 0, "step": 50, "loss": 0.0, "loss_parts": {}, "lr": 0.0, "time": 0.0}
{"type": "epoch", "epoch": 0, "train_loss": 0.0, "val": {"epe": {}, "OA": 0, "MCA": 0}, "time": 0.0}
```

- 步间隔放配置里：`log.interval=50`。
- 启动命令把 stdout 与 stderr 都写进 `stdout.log`。
- resume 时续写同一个文件，并先写一行 `{"type": "resume", "step": ...}`。
- tensorboard 记录与 `train.jsonl` 相同的标量，不额外加别的。
- 评测日志放 `$RESULTS/<round>/eval/<exp_id>/<run_id>/eval.log`。
- `$CACHE/cmdlogs/` 是 agent 看命令输出的缓存，不是档案，报告和 registry 不引用它。

## 结果追溯

- `metrics.json` 的 `meta` 含：run_id，ckpt 路径，git 哈希，配置哈希，数据卡哈希。
- 工作区有未提交改动时，`run_manifest.json` 记 `git_dirty: true`，并把 `git diff` 存成 run 目录下的 `code.diff`。
- 结果表由 `scripts/make_table.py` 生成。输入是 run_id 列表，输出是 markdown。文件头写生成命令，git 哈希，run_id 列表。
- 表里放上面三行参考值，注明"转引，未核对"。
- 报告里的每个数字，都能顺着"表，run_id，metrics.json"找到来源。结论句指向表，不直接写数。

## registry

- `$RESULTS/registry.jsonl` 是唯一的真源。一行一个 run：run_id，exp_id，配置，种子，状态，三档 EPE，实验区路径，结果区路径，git 哈希，起止时间。
- 状态取 `running`，`done`，`failed`。只追加，不改旧行。同一个 run_id 以最后一行为准。
- 由 `scripts/registry.py` 在训练开始，训练结束，评测结束时写入。
- `docs/experiments/registry.md` 由 `scripts/registry.py --md` 从 jsonl 生成，文件头写"自动生成，不手改"。
