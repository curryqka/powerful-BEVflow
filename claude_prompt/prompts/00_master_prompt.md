# BEVflow 集群部署任务：总控提示词

把本文件整体作为 agent 的第一条消息。建好仓库后，把第 0 节和第 3 节原样写入仓库根的 CLAUDE.md，之后每个会话先读它。

## 0. 变量

要改路径，只改这一节。

```
PROJECT = bevflow
CODE    = /home/kaneqian/BEVflow
NAS     = /media/nas_users/kaneqian
EXP     = $NAS/Experiment/bevflow
RESULTS = $NAS/results/bevflow
DATA    = $NAS/data/bevflow
CACHE   = $EXP/_cache
RAW_NUSCENES = /media/nas_users/kaneqian/data/Nuscences
```

## 1. 背景与目标

我在做 BEV 类别无关运动场预测的论文，目标是 IROS 2027。输入是过去 5 帧 LiDAR，输出是每个 BEV 格子未来 1 秒的位移。基线谱系是 MotionNet、BE-STI、PriorMotion、LEGO-Motion。后两个是我自己的工作，我熟悉这个方向，你可以自主推进，不用每步等我确认。

基线现状：

- MotionNet 只有官方权重，需要下载。
- PriorMotion 和 LEGO-Motion 没有现成权重。要靠逆向复现，再训练出来。
- BE-STI 没有官方代码，只引论文数字，除非我另有要求。

这一轮建地基，不做方法创新。四个出口：

1. 在集群上跑通 MotionNet 的数据生产，训练，评测，可视化。
2. 把代码重构成清晰的模块，让后面的新方法只改一处。
3. 两个存储区各司其职，不污染其他项目。
4. 固定一套和 MotionNet 对齐的经验参数，拿到第一轮结果。

逆向复现 PriorMotion 和 LEGO-Motion 是第六阶段，我触发后再做。

## 2. 两个存储区

| 区域 | 路径 | 放什么 |
|---|---|---|
| 代码区 | `$CODE` | 代码，配置，简要 docs |
| 实验区 | `$EXP` | 训练 run：ckpt，训练日志，配置快照，uv 环境，命令日志缓存 |
| 结果区 | `$RESULTS` | 评测结果，可视化，结果表 |
| 数据区 | `$DATA` | 预处理数据，split 文件，数据卡。原始 nuScenes 只放软链接 |

每个根目录下，本项目只用名为 `bevflow` 的二级目录。别的二级目录属于其他项目，不读不写。详见 skill `bevflow-workspace-layout`。

## 3. 规则

### 边界

1. 只写 `$CODE`，`$EXP`，`$RESULTS`，`$DATA`。
2. `/home` 不放大文件。代码区里的 `data`，`experiment`，`results` 是软链接，写进 `.gitignore`。
3. 删除文件要敬畏。除了删你自己生成的临时文件，其他删除都先问我。脚本里的删除，默认只列出，必须显式 flag 才真删。
4. `mv`，`cp`，`rm` 的批量或带逻辑的操作，先打印将要执行的命令，确认无误再执行。
5. `third_party/MotionNet` 是上游，只读。要改，写成 `patches/` 下的补丁，并注明原因。
6. 每个数字来自日志或 `metrics.json`，不凭记忆写，不编造。
7. 不用 sudo。共享集群上，登录节点不跑重活，占卡前先看 `nvidia-smi`。

### 不确定就问我

不确定的地方多问，一次问一批。判断标准：这个选择会影响结果的解释，或者事后难以回退，就算不确定。例如 split 的来源，统一用哪套数据管线，上游代码里语义不清的参数。常规的工程决定自己定，在文档里记一行。

问的同时，继续做不依赖答案的部分。

### 代码规范

1. 注释：一眼能懂的代码不写注释。只有跨越多行才能看懂的逻辑才写。
2. docstring：不过度写。多数函数不写，更不要写会抛什么异常。
3. 异常：输入有预期，且上游逻辑已保证的，用 `assert`。严重且无法恢复的错误，不能 catch，要让它中断程序。禁止 `try: ... except Exception as e: print(...)` 这类写法。
4. 删除：见上面的边界第 3 条。

### 命令输出

输出预计很长的命令，先重定向到 `$CACHE/cmdlogs/<name>.log`，再用 `head` 或 `tail` 查看。不要为了看输出而重复运行命令。

## 4. 阶段

每个阶段开始前，先读对应的 skill。检查点通过就自动进入下一阶段，不用等我确认。每个检查点提交一次 git。

### P0 勘察，只读

查调度器，GPU，磁盘，uv，网络出口，原始 nuScenes 的位置。结果写 `docs/setup/recon.md`。
检查点：调度器，可用 GPU，磁盘空间，能否访问 github 与 zenodo，原始 nuScenes 路径，uv 版本，这几项都有答案。

### P1 地基

skill：`bevflow-workspace-layout`，`bevflow-env-setup`。
建目录与仓库，克隆并固定上游，用 uv 建环境，把数据软链接到上游期望的结构。下载 MotionNet 官方权重（Zenodo 8215963），记录文件名，大小与哈希。
检查点：目录齐全，上游模块能 import，nuScenes 元数据能加载，官方权重已下载并记录哈希。

### P2 小批量冒烟，用上游原始代码

skill：`bevflow-data-pipeline`，`bevflow-smoke-test`。
在 nuScenes mini 上走通数据生产，训练，存取 ckpt，评测，可视化，并实测 bs4 与 bs2 的显存与耗时。
检查点：`scripts/smoke.sh` 五步通过。

### P3 重构

skill：`bevflow-refactor`。
按五个模块组织：数据预处理，数据加载器，模型网络，训练器，评测器。外加实验配置和分类文档。
检查点：行为不变的测试通过，新结构上 `smoke.sh` 通过。

### P4 固定参数与第一轮结果

skill：`bevflow-experiment-run`，`bevflow-data-pipeline`。
提取参数卡。全量预处理。用我们的评测器评测 MotionNet 官方权重（P1 已下载）。按对齐参数训练 R1A 与 R1B。评测，可视化，写报告。
检查点：结果表和报告已写好，与参考值的偏差已排查。

### P5 收尾

写 `docs/experiments/howto.md`，一条命令能复现全流程。写 P6 的准备清单。然后停下，等我触发。

### P6 逆向复现，我触发后才做

skill：`bevflow-reverse-reproduce`。我会给出旧项目代码的路径。

## 5. 工作方式

先用一句话告诉我你要做什么，然后开工。

- 默认自己推进到 P5，中途不停下来等批准。
- 长任务提交到调度器，或用 nohup 加 tmux。提交后去做不依赖它的事，不要空等。
- 每个阶段结束，用三到五句话汇报：做了什么，下一步，需要我拍板的事。
- 文档简短，分类放在 `docs/`，见 skill `bevflow-docs-report`。

## 6. 本轮不做

- 不做方法创新，包括流匹配解码器，零流锚定，位姿残差。
- 不改上游算法逻辑。重构只改结构。
- 不装系统级软件。

## 7. 完成的定义

1. P0 到 P5 的检查点都通过。
2. 一条命令能复现数据生产，训练，评测，可视化。
3. `docs/reports/R1.md` 给出结果表，参考值，偏差与原因。
4. P6 的准备清单已写好。
