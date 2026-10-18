---
name: bevflow-workspace-layout
description: 在集群上创建、检查或使用 BEVflow 的目录与路径时使用。覆盖代码区与实验区的分工，项目二级目录隔离，软链接，路径配置，run 命名与产物清单。
---

# 目标

代码区只放代码和简要 docs，大文件全部在 NAS。本项目只占用每个根下名为 `bevflow` 的二级目录。

# 变量

沿用总控提示词第 0 节的 `CODE`，`NAS`，`EXP`，`RESULTS`，`DATA`，`CACHE`。

# 目录树

```
$CODE/                          git 仓库
├── bevflow/
│   ├── preprocess/             数据预处理
│   ├── datasets/               数据加载器
│   ├── models/                 模型网络，含损失
│   ├── trainers/               训练器
│   ├── evaluators/             评测器，含指标与可视化
│   └── utils/                  paths，seed，config，registry
├── configs/                    base，data，model，exp
├── scripts/                    薄 CLI 入口，slurm 模板，smoke.sh
├── tests/
├── patches/                    对上游的补丁
├── third_party/MotionNet/      上游，只读
├── ref/<name>/                 P6 的论文 PDF 与旧代码，只读，gitignore，不开源
├── docs/                       setup，design，experiments，reports，reference
├── data        -> $DATA        软链接，gitignore
├── experiment  -> $EXP         软链接，gitignore
└── results     -> $RESULTS     软链接，gitignore

$EXP/
├── _env/bevflow                uv 虚拟环境
├── _cache/                     cmdlogs/  uv/
├── _data_logs/                 预处理日志
└── <exp_id>/<run_id>/          ckpt/  logs/  config_snapshot.yaml  run_manifest.json
                                logs/ 的内容见 skill bevflow-experiment-run 第七节

$RESULTS/
├── <round>/eval/<exp_id>/<run_id>/     metrics.json  per_ckpt.json
├── <round>/viz/<exp_id>/<run_id>/      图片与 viz_manifest.json
├── <round>/tables/
├── _smoke/<timestamp>/
└── registry.jsonl                      每个 run 一行

$DATA/
├── raw/nuscenes  -> 已有的原始数据     只读软链接，不复制
├── splits/
└── preprocessed/<data_version>/{train,val,test}/    含 data_card.json
```

实验区放训练过程产物，结果区放评测与展示产物。这是默认约定，可改。

# 创建

1. 对每个 NAS 根，`ls` 看二级目录名，只确认 `bevflow` 在不在，不进入别的项目。
2. `mkdir -p` 建树。已存在就复用。
3. 在 `$CODE` 里建三个软链接，用绝对路径，用 `readlink -f` 确认。把软链接和 `ref/` 写进 `.gitignore`。

# 路径来源只有一个

代码里不写绝对路径。所有路径经过 `bevflow/utils/paths.py`。它读环境变量 `BEVFLOW_CODE`，`BEVFLOW_EXP`，`BEVFLOW_RESULTS`，`BEVFLOW_DATA`，缺省值取自 `configs/paths.yaml`。

写文件的入口先 `assert` 目标路径的 `realpath` 在这四个根之下。

# run 命名与清单

- `exp_id` 沿用实验卡编号，如 `R1A`。
- `run_id` 为 `YYYYMMDD_<exp_id>_<model>_s<seed>`。
- 每个 run 写 `run_manifest.json`：git 哈希，是否有未提交改动，配置哈希，数据卡哈希，种子，主机，GPU，起止时间，启动命令。
- `registry.jsonl` 由脚本写入，规则见 skill `bevflow-experiment-run` 第七节。

# NAS 小文件

NAS 读小文件慢时，把当前 job 要用的数据复制到本地盘训练，训练完丢弃。永久位置仍然是 `$DATA`。
