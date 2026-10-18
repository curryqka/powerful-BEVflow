# BEVflow 集群部署包（总控提示词 + P6 指挥手册 + 九个 skill）

> 合订本，便于在项目里检索。可直接部署的目录版见 zip。每个文件前的路径是它在包里的位置。本版新增：P6 以论文为规格，P7 库化与开源，P6 指挥手册。

---

<!-- FILE: README.md -->

## 文件：`README.md`

### BEVflow 集群部署包：使用说明

#### 内容

```
prompts/00_master_prompt.md           总控提示词，贴给 agent 当第一条消息
prompts/10_p6_command_handbook.md     P6 指挥手册，给你自己用，不是给 agent 读的
skills/bevflow-library/               库化与开源：接口约定，包布局，model zoo，许可审计
skills/bevflow-workspace-layout/      目录与路径，两个存储区，隔离
skills/bevflow-env-setup/             勘察，上游仓库，uv 环境，补丁协议
skills/bevflow-data-pipeline/         nuScenes 链接，划分，数据生产，数据卡
skills/bevflow-smoke-test/            五步冒烟，可视化数值检查
skills/bevflow-refactor/              五模块重构，行为不变的测试
skills/bevflow-experiment-run/        参数卡，两套配置，官方权重，第一轮
skills/bevflow-reverse-reproduce/     第六阶段：论文为规格，旧代码为参考，你触发后才用
skills/bevflow-docs-report/           文档分类，报告模板
```

#### 部署

1. 把 `skills/` 下的九个目录复制到集群的 `~/.claude/skills/`。已部署过的，只需要新增 `bevflow-library`，并覆盖 `bevflow-reverse-reproduce` 与总控提示词。仓库建好后，也可以放进 `$CODE/.claude/skills/`。
2. 把 `prompts/00_master_prompt.md` 整体贴给 agent，作为第一条消息。
3. 让 agent 建好仓库后，把总控提示词的第 0 节与第 3 节写进仓库根的 `CLAUDE.md`。第 3 节含你的开发规范，每个会话都会重新读到。

#### 这一版的取向

- 自动推进。检查点通过就进入下一阶段，不等确认。agent 默认一路做到 P5。
- 只在"会影响结果解释，或事后难以回退"的事情上提问，一次问一批。常规工程决定自己定。
- 参考值对照是对照，不是门。对不上就排查并写进报告，不卡住后面的工作。
- 环境用 uv：`uv venv` 加 `uv pip install`，前缀与缓存都在 `$EXP` 下。
- 基线现状按实际写：只有 MotionNet 官方权重，需要下载。PriorMotion 与 LEGO-Motion 在 P6 复现后训练。BE-STI 只引论文数字。

#### P6 与 P7 的取向

- 论文是规格，旧代码是参考实现。旧代码的结构与数据加载都当作可能有错，要审计。
- 冲突不静默选择，进决策清单，由你批量裁决。
- 先设计接口，你批准后冻结，再写模型。
- 数据层先行，数据测试通过才做模型。
- 许可与出处审计在开源前必做。MERL 版 MotionNet 是 AGPL，派生代码的许可要兼容，决定权在你。

用法见 `prompts/10_p6_command_handbook.md`。

#### 开发规范在哪里

全部在总控提示词第 3 节：

- 注释，docstring，异常（assert 与不 catch 致命错误）。
- 删除文件需确认，脚本里的删除默认只列出，必须显式 flag。
- 长输出先写入 `$CACHE/cmdlogs/` 再 head 或 tail。
- `mv`，`cp`，`rm` 的批量操作，先打印命令再执行。

"使用 Claude 的要求"那条，不要直接 accept edits 去做复杂文件操作，是给你自己的操作习惯，agent 不需要执行。agent 侧对应的是上面最后一条。

#### 默认值，可改

| 项目 | 默认 | 改哪里 |
|---|---|---|
| 项目二级目录名 | `bevflow` | 总控提示词第 0 节 |
| 代码区 | `/home/kaneqian/BEVflow` | 同上 |
| NAS 根 | `/media/nas_users/kaneqian` | 同上 |
| 实验区与结果区的分工 | 实验区放训练 run，结果区放评测与可视化 | skill `bevflow-workspace-layout` |
| uv 环境位置 | `$EXP/_env/bevflow` | skill `bevflow-env-setup` |
| 上游仓库 | MERL 版 MotionNet，克隆后与作者页对照 | skill `bevflow-env-setup` |

#### 你可能要提供的输入

| 输入 | 什么时候要 |
|---|---|
| 原始 nuScenes 的路径 | P1，找不到时 agent 会问 |
| 旧管线的 500/100/250 split 文件 | P1，没有就退回上游自带划分 |
| 调度器分区与账号 | P0，需要时 agent 会问 |
| 论文 PDF，旧代码，`notes.md`，可用的日志与 ckpt | P6，放进 `ref/<name>/` |
| 论文数字对应的评测口径 | P6 的 C1 阶段 |
| 库名与许可方案 | P7 |

---

<!-- FILE: prompts/00_master_prompt.md -->

## 文件：`prompts/00_master_prompt.md`

### BEVflow 集群部署任务：总控提示词

把本文件整体作为 agent 的第一条消息。建好仓库后，把第 0 节和第 3 节原样写入仓库根的 CLAUDE.md，之后每个会话先读它。

#### 0. 变量

要改路径，只改这一节。

```
PROJECT = bevflow
CODE    = /home/kaneqian/BEVflow
NAS     = /media/nas_users/kaneqian
EXP     = $NAS/Experiment/bevflow
RESULTS = $NAS/results/bevflow
DATA    = $NAS/data/bevflow
CACHE   = $EXP/_cache
```

#### 1. 背景与目标

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

逆向复现 PriorMotion 和 LEGO-Motion 是第六阶段，库化与开源准备是第七阶段，都等我触发。最终目的是把我之前的工作和后续的新方法整理成一个开源算法库，所以从现在起，代码要按"别人也要用"来写。

#### 2. 两个存储区

| 区域 | 路径 | 放什么 |
|---|---|---|
| 代码区 | `$CODE` | 代码，配置，简要 docs |
| 实验区 | `$EXP` | 训练 run：ckpt，训练日志，配置快照，uv 环境，命令日志缓存 |
| 结果区 | `$RESULTS` | 评测结果，可视化，结果表 |
| 数据区 | `$DATA` | 预处理数据，split 文件，数据卡。原始 nuScenes 只放软链接 |

每个根目录下，本项目只用名为 `bevflow` 的二级目录。别的二级目录属于其他项目，不读不写。详见 skill `bevflow-workspace-layout`。

#### 3. 规则

##### 边界

1. 只写 `$CODE`，`$EXP`，`$RESULTS`，`$DATA`。
2. `/home` 不放大文件。代码区里的 `data`，`experiment`，`results` 是软链接，写进 `.gitignore`。
3. 删除文件要敬畏。除了删你自己生成的临时文件，其他删除都先问我。脚本里的删除，默认只列出，必须显式 flag 才真删。
4. `mv`，`cp`，`rm` 的批量或带逻辑的操作，先打印将要执行的命令，确认无误再执行。
5. `third_party/MotionNet` 是上游，只读。要改，写成 `patches/` 下的补丁，并注明原因。
6. 每个数字来自日志或 `metrics.json`，不凭记忆写，不编造。
7. 不用 sudo。共享集群上，登录节点不跑重活，占卡前先看 `nvidia-smi`。

##### 不确定就问我

不确定的地方多问，一次问一批。判断标准：这个选择会影响结果的解释，或者事后难以回退，就算不确定。例如 split 的来源，统一用哪套数据管线，上游代码里语义不清的参数。常规的工程决定自己定，在文档里记一行。

问的同时，继续做不依赖答案的部分。

##### 代码规范

1. 注释：一眼能懂的代码不写注释。只有跨越多行才能看懂的逻辑才写。
2. docstring：不过度写。多数函数不写，更不要写会抛什么异常。
3. 异常：输入有预期，且上游逻辑已保证的，用 `assert`。严重且无法恢复的错误，不能 catch，要让它中断程序。禁止 `try: ... except Exception as e: print(...)` 这类写法。
4. 删除：见上面的边界第 3 条。

##### 命令输出

输出预计很长的命令，先重定向到 `$CACHE/cmdlogs/<name>.log`，再用 `head` 或 `tail` 查看。不要为了看输出而重复运行命令。

#### 4. 阶段

每个阶段开始前，先读对应的 skill。检查点通过就自动进入下一阶段，不用等我确认。每个检查点提交一次 git。

##### P0 勘察，只读

查调度器，GPU，磁盘，uv，网络出口，原始 nuScenes 的位置。结果写 `docs/setup/recon.md`。
检查点：调度器，可用 GPU，磁盘空间，能否访问 github 与 zenodo，原始 nuScenes 路径，uv 版本，这几项都有答案。

##### P1 地基

skill：`bevflow-workspace-layout`，`bevflow-env-setup`。
建目录与仓库，克隆并固定上游，用 uv 建环境，把数据软链接到上游期望的结构。
检查点：目录齐全，上游模块能 import，nuScenes 元数据能加载。

##### P2 小批量冒烟，用上游原始代码

skill：`bevflow-data-pipeline`，`bevflow-smoke-test`。
在 nuScenes mini 上走通数据生产，训练，存取 ckpt，评测，可视化，并实测 bs4 与 bs2 的显存与耗时。
检查点：`scripts/smoke.sh` 五步通过。

##### P3 重构

skill：`bevflow-refactor`。
按五个模块组织：数据预处理，数据加载器，模型网络，训练器，评测器。外加实验配置和分类文档。
检查点：行为不变的测试通过，新结构上 `smoke.sh` 通过。

##### P4 固定参数与第一轮结果

skill：`bevflow-experiment-run`，`bevflow-data-pipeline`。
提取参数卡。全量预处理。下载 MotionNet 官方权重并用我们的评测器评测。按对齐参数训练 R1A 与 R1B。评测，可视化，写报告。
检查点：结果表和报告已写好，与参考值的偏差已排查。

##### P5 收尾

写 `docs/experiments/howto.md`，一条命令能复现全流程。写 P6 的准备清单。然后停下，等我触发。

##### P6 逆向复现，我触发后才做

skill：`bevflow-reverse-reproduce`，`bevflow-library`。
材料放在 `ref/<name>/`：论文 PDF，旧代码，我的备注。规格以论文为准，旧代码只是参考实现，结构与数据加载都可能有错，要审计，不要继承。论文与旧代码冲突，写进决策清单等我裁决，不要静默选择。先做数据层，再做模型。
检查点：规格文档，审计文档，决策记录齐全。数据层测试通过。复现的结果与论文数字的对照表已写好，偏差已排查。

##### P7 库化与开源准备，我触发后才做

skill：`bevflow-library`。
整理接口，包布局，model zoo，出处与许可审计，私有信息清理，CPU 单元测试，README。许可方案由我决定。不公开仓库，不上传权重。

#### 5. 工作方式

先用一句话告诉我你要做什么，然后开工。

- 默认自己推进到 P5，中途不停下来等批准。
- 长任务提交到调度器，或用 nohup 加 tmux。提交后去做不依赖它的事，不要空等。
- 每个阶段结束，用三到五句话汇报：做了什么，下一步，需要我拍板的事。
- 文档简短，分类放在 `docs/`，见 skill `bevflow-docs-report`。

#### 6. 本轮不做

- 不做方法创新，包括流匹配解码器，零流锚定，位姿残差。
- 不改上游算法逻辑。重构只改结构。
- 不装系统级软件。

#### 7. 完成的定义

1. P0 到 P5 的检查点都通过。
2. 一条命令能复现数据生产，训练，评测，可视化。
3. `docs/reports/R1.md` 给出结果表，参考值，偏差与原因。
4. P6 的准备清单已写好。

---

<!-- FILE: prompts/10_p6_command_handbook.md -->

## 文件：`prompts/10_p6_command_handbook.md`

### P6 指挥手册：复现 PriorMotion 与 LEGO-Motion，并库化

这份手册给你用，不是给 agent 读的。每条指令可以直接贴给 agent。`<name>` 换成 `priormotion` 或 `legomotion`。

#### 总流程

```
材料就位
  C0 摸底
  C1 规格提取        你审：规格与论文是否一致
  C2 旧代码审计      你审：不一致与缺失项
  C3 决策裁决        你答：批量回复 D1 到 Dn
  C4 库设计          你批：接口文档，之后冻结
  C5 数据层          你审：数据测试结果
  C6 模型移植        一次一个模块
  C7 训练与对照      你审：对照表与偏差
  C8 并行            两个模型分支
  C9 库化与开源准备  你决：许可方案
```

最重要的两个审阅点是 C3 和 C4。决策和接口一旦定下，后面的返工成本最低。

#### 材料就位

让 agent 或你自己把材料放成这样：

```
$CODE/ref/<name>/
├── paper.pdf
├── code/
└── notes.md
```

`notes.md` 里写：旧代码对应论文的哪个版本，你已知的问题，哪些部分你自己都不放心，旧项目里还有没有日志与 ckpt，论文数字对应哪套评测。写得越具体，后面问你的越少。

#### 指令模板

##### C0 摸底

> 读 `ref/<name>/` 下的材料。告诉我：论文页数，有哪些表和公式，旧代码的规模与入口，缺什么材料。不要改任何东西。

##### C1 规格提取

> 用 skill `bevflow-reverse-reproduce` 的第 1 步，为 `<name>` 写 `docs/design/spec_<name>.md`。每条标页码、公式号或表号。读不懂的单列歧义清单。公式乱码处标待确认，不要猜。

你审：报告数字表是否抄对，网络结构的张量形状是否与你的记忆一致，歧义清单里哪些你能当场回答。

##### C2 旧代码审计

> 对照 `spec_<name>.md` 审计 `ref/<name>/code/`，写 `audit_<name>.md`。先做追溯表，再做数据层专项检查。不要修改旧代码。

你审：状态为"不一致"和"缺失"的条目，以及数据层专项里每一项的结论。

##### C3 决策裁决

> 把审计里所有不一致与歧义整理成 D1 到 Dn，写进 `decisions_<name>.md`，每条给现象，证据，选项，建议。整理完在聊天里列出清单。等我回复期间，继续做不依赖答案的部分。

你答的格式，越短越好：

```
D1: A
D2: 按论文
D3: 旧代码是对的，因为 ...
D4: 不确定，你先按论文做，记下来
```

##### C4 库设计

> 用 skill `bevflow-library` 的第一、二节，写 `docs/design/interfaces.md` 草案和包布局。读完 MotionNet 上游，旧代码和 spec 之后再设计。列出共享部件清单。设计放在文档里，先不写模型代码。

你批：batch 与输出 dict 的键，`predict` 与 `sample` 的约定，哪些东西进 `common`。批准后告诉 agent"接口冻结"。

##### C5 数据层

> 按 `decisions_<name>.md` 实现数据层，先于模型。额外产物生成到新版本目录并写数据卡。跑 skill 第 5 步里的数据层测试，把结果汇总给我。

你审：与 `motionnet_v1` 的逐样本一致性，标签三档占比，自车补偿检查。

##### C6 模型移植

> 移植 `<name>` 的 `<模块名>`。按 spec 实现。旧代码可跑且审计认为该模块与论文一致时，做同权重同输入的对比。做完汇报形状与数值检查，然后停下，等我说下一个模块。

顺序：共享主干，输出头，损失，训练调度。想放手让它连着做，就改成"按顺序做完全部模块，每个模块过了测试再做下一个"。

##### C7 训练与对照

> 按 skill `bevflow-reverse-reproduce` 的第 7 步，对 `<name>` 做短训练，再做完整训练。评测并生成对照表，含论文数字，参数量，延迟，消融趋势。对不上时按顺序排查，记进 `reproduce_<name>.md`。不要为了对齐数字调参。

你审：偏差大的项，排查记录里 agent 的判断是否站得住。

##### C8 并行

> `common` 部分已合入主分支。为 `priormotion` 与 `legomotion` 各开一个 git worktree 分支，分别推进。合并前跑全部测试。共享部分有冲突时，停下问我。

##### C9 库化与开源准备

> 用 skill `bevflow-library` 做库化。先写出处审计 `provenance.md`，列出许可方案选项，不要替我选。然后做私有信息扫描，CPU 单元测试，README，model zoo。不要公开仓库，不要上传权重。

你决：许可方案。这是唯一必须由你来做的决定。

#### 纠偏与追问用语

| 想做什么 | 这样说 |
|---|---|
| 阻止它擅自选择 | "这是论文与旧代码的冲突，写进决策清单，不要自己选。" |
| 要证据 | "给我论文页码或旧代码行号。" |
| 收窄范围 | "只改数据层，不动模型。" |
| 防止顺手改进 | "只复现，不改进。发现可改进的点记在 `docs/design/ideas.md`。" |
| 怀疑旧代码 | "不要把旧代码当真值，按论文核对这一段。" |
| 对不上数字时 | "先查数据，再查损失权重，再查模块，每一步写进排查记录。" |
| 需要停下 | "做到这里，汇报，等我。" |
| 它读不了图 | "架构图我口述，你照着写。" |

#### 几个提醒

- 论文数字对应的评测口径，要在 C1 阶段就和你确认，否则后面所有对照都不可信。
- 旧代码里"能跑出好结果"的 trick，如果论文没写，agent 会列进决策清单。这类条目要你亲自判断，是论文遗漏，还是后来加的。
- 并行之前，一定要先把 `common` 合入主分支。
- 开源动作本身，推送公共仓库，上传权重，都由你来做。

---

<!-- FILE: skills/bevflow-data-pipeline/SKILL.md -->

## 文件：`skills/bevflow-data-pipeline/SKILL.md`

---
name: bevflow-data-pipeline
description: 处理 BEVflow 的数据时使用。覆盖原始 nuScenes 的软链接，500/100/250 划分，MotionNet 数据生产，数据卡，统计，以及 mini 冒烟数据。
---

### 目标

把 nuScenes 变成 MotionNet 协议的预处理数据，放进 `$DATA`，并能追溯到参数，划分和上游版本。

### 一、原始数据只链接

1. 找原始 nuScenes：先看环境变量与 `docs/setup/recon.md`，再看 `$NAS/data` 的一级目录名，最后在常见位置浅层找，深度不超过 3。
2. 确认有 `v1.0-trainval` 或 `v1.0-mini` 元数据，以及 `samples`，`sweeps`。
3. 建链接 `$DATA/raw/nuscenes -> <原始路径>`，只读使用。
4. 找不到，问我。不要全盘搜索 NAS。

### 二、划分

目标是 500/100/250 个场景，MotionNet 协议。来源按优先级：

1. 我旧管线里的 split 文件，先问我要路径。
2. 上游 `gen_data.py` 自带的划分，读 `--help` 与源码确认。
3. MotionNet 论文附录。

存到 `$DATA/splits/`，并 `assert` 三个数：train 500，val 100，test 250，且三者无交集。断言不过，说明来源不对，问我，不要自己改数。

冒烟用 mini 划分：10 个场景分成 6，2，2。

### 三、数据生产

1. 用上游的数据生成脚本。参数从上游代码读，不凭记忆。
2. 输出到 `$DATA/preprocessed/<data_version>/{train,val,test}`。正式版本叫 `motionnet_v1`，mini 版本叫 `mini`。
3. 先在 mini 上跑通并计时，估计全量耗时。耗时长，就用调度器的 array job 按场景分片。
4. 任务可恢复：已生成的样本跳过。
5. 日志写到 `$EXP/_data_logs/`。

### 四、数据卡

每个预处理版本下写 `data_card.json`：

```json
{
  "data_version": "motionnet_v1",
  "upstream_commit": "...",
  "split_file": "...",
  "params": {"voxel_size": null, "range": null, "height_bins": null,
             "past_frames": null, "future_frames": null},
  "counts": {"train": 0, "val": 0, "test": 0}
}
```

`params` 的值从上游代码读出，注明来源文件与行号。数据加载器启动时读数据卡，参数与配置不一致就 `assert` 失败。

### 五、软链接到上游期望的目录

读上游训练脚本，找出它期望的数据路径。在 `$CODE` 下建对应的软链接，指向 `$DATA/preprocessed/<data_version>`。不改上游的路径默认值。

### 六、统计

写 `scripts/check_data.py`，对每个 split：

- 随机抽 20 个样本，检查形状，dtype，无 NaN 与 Inf。
- 统计 GT 速度三档占比：静态，不超过 5 m/s，超过 5 m/s。
- 统计动态格子占非空格子的比例。后面评估稀疏生成的收益要用它，现在顺手记下。

结果写进 `data_stats.json`。

---

<!-- FILE: skills/bevflow-docs-report/SKILL.md -->

## 文件：`skills/bevflow-docs-report/SKILL.md`

---
name: bevflow-docs-report
description: 写 BEVflow 仓库里的 docs 和实验报告时使用。覆盖文档分类结构，报告模板，简要写作规则。
---

### 目标

文档简短，分类存放，读者一分钟内找到结论。`$CODE` 里只放简要 docs，长日志和大表放 `$RESULTS`，文档里给路径。

### 分类结构

```
docs/
├── README.md        索引，一句话说明每个子目录
├── setup/           recon  upstream  upstream_patches  env  resources  smoke
├── design/          architecture  upstream_map  interfaces  reproduce_<name>
├── experiments/     howto  registry  protocols
├── reports/         R1  reproduce_<name>
└── reference/       param_card  baseline_numbers
```

每篇不超过两页。超过就拆。

### 报告模板

1. 目的：这次回答什么问题，一句话。
2. 设置：数据版本，划分，配置文件，参数卡链接，种子，硬件。
3. 结果：一张表。数字来自 `metrics.json`，脚本生成。表里放参考值，注明来源与"转引，未核对"。
4. 偏差：与参考值的偏差，以及已排查的原因。
5. 结论：测出来的写"我们观察到"，没测的写"我们假设"并标【假设】。
6. 局限与下一步。

### 写作

若装有 `kaiming-style` skill，先加载它，以它为准。没有就按这几条：

- 问题优先，一句话讲清核心，再展开。
- 短句，主动语态，一句不超过 40 字。
- 用具体的词和数字，不写"显著""强大"。
- 事实和假设分开。
- 同一个概念全文只用一个名字。
- 每段第一句是结论。
- 不用破折号和分号。括号，顿号，冒号都克制。

---

<!-- FILE: skills/bevflow-env-setup/SKILL.md -->

## 文件：`skills/bevflow-env-setup/SKILL.md`

---
name: bevflow-env-setup
description: 在集群上勘察机器，并用 uv 按 MotionNet 上游仓库配置 Python 环境时使用。覆盖勘察，克隆并固定上游，uv 虚拟环境，补丁协议，验证与锁文件。
---

### 目标

得到一个能跑 MotionNet 的 uv 环境，别人能复现。上游代码保持原样，改动写成补丁。

### 一、勘察，只读

```bash
hostname; nvidia-smi; nvcc --version 2>/dev/null
uv --version; python3 --version
command -v sinfo squeue sbatch && sinfo -o "%P %a %l %D %G" | head -20
df -h "$HOME" "$NAS"
curl -sI --max-time 10 https://github.com | head -1
curl -sI --max-time 10 https://zenodo.org | head -1
```

输出整理进 `docs/setup/recon.md`。要回答的问题：

- 有调度器吗？分区，时限，GPU 类型。没有就记下哪些卡空闲。
- GPU 计算能力是多少？它决定 torch 与 CUDA 版本。
- 网络通哪些站点？不通，就列出要我上传的文件。
- uv 是否已装？没装就问我，不要自己装到别处。

### 二、克隆并固定上游

1. 克隆 `https://github.com/merlresearch/MotionNet` 到 `third_party/MotionNet`。
2. 项目笔记说作者页 `pxiangwu/MotionNet` 只有 README。克隆后两个仓库都看一眼，以含数据生成，训练，评测脚本的为准。
3. 记录 commit，license，README 的环境要求，以及这几件事对应的脚本：生成数据，训练 STPN 主模型，评测位移，评测分类。写进 `docs/setup/upstream.md`。

克隆不通就问我，不要换镜像域名。

### 三、用 uv 建环境

环境前缀放 `$EXP/_env/bevflow`。uv 缓存放 `$EXP/_cache/uv`，避免占满 `$HOME`。

```bash
export UV_CACHE_DIR=$EXP/_cache/uv
uv venv $EXP/_env/bevflow --python 3.10
source $EXP/_env/bevflow/bin/activate
```

安装用 `uv pip install`：

1. 先装 torch。版本由 GPU 计算能力和驱动决定，不照搬上游的老版本。较新的 uv 支持 `--torch-backend=auto`，先用 `uv pip install --help` 确认。不支持就指定 `--index-url https://download.pytorch.org/whl/cuXXX`。
2. 再装 numpy 等科学计算库，nuscenes-devkit，以及上游 requirements 里的其余包。
3. 每装一步就验证，不要最后一起验证。

### 四、补丁协议

- 不直接改上游目录。需要改时，用 `diff -u` 生成 `patches/NNNN-<name>.patch`，在 `docs/setup/upstream_patches.md` 记一行：改了什么，为什么。
- 常见需要补丁的地方：`np.int` 与 `np.float` 已移除，`torch.load` 默认参数变化，过时的 DataLoader 参数，路径硬编码。
- 补丁只修兼容，不改算法。拿不准算不算算法改动，问我。

### 五、验证

```bash
python - <<'EOF'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda"); print((x @ x).sum().item())
import nuscenes; print("nuscenes ok")
EOF
```

再 import 上游的模型与数据模块，并用 mini 元数据实例化一次 `NuScenes`。

### 六、锁文件

提交 `requirements.lock.txt`，由 `uv pip freeze` 生成，并在 `docs/setup/env.md` 写下重建环境的命令。

遇到装不上的包，先看报错，再决定换版本还是写补丁。判断不了就问我。

---

<!-- FILE: skills/bevflow-experiment-run/SKILL.md -->

## 文件：`skills/bevflow-experiment-run/SKILL.md`

---
name: bevflow-experiment-run
description: 固定与 MotionNet 对齐的经验参数并跑第一轮实验时使用。覆盖参数卡，两套配置，MotionNet 官方权重下载与评测，训练运行，ckpt 保存，参考值对照，结果登记。
---

### 目标

得到第一轮可信的结果：参数有出处，数字来自日志，偏差有解释。

### 一、参数卡

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

### 二、两套配置

| 配置 | 用途 | 内容 |
|---|---|---|
| `exp/R1A_official.yaml` | 验证管线 | 上游默认训练配方 |
| `exp/R1B_budget.yaml` | 项目协议 | 数据与模型同 R1A，训练预算改为 45 epoch，bs4 或 bs2 加累积 2 |

R1B 之后是所有新方法共用的预算。两套配置只在训练预算上不同。

### 三、流程

1. **下载 MotionNet 官方权重。** Zenodo 记录号 8215963。先打开页面确认文件名与大小，下载后记录哈希。网络不通就让我上传。
2. **评测官方权重。** 用我们的评测器在 test 上评测，这是整条管线的第一个对照点。
3. **训练 R1A。** 种子 0。评测最后 10 个 epoch 的 ckpt，写 `per_ckpt.json`。
4. **可视化。** 官方权重与 R1A 各出一套，样本固定，见 skill `bevflow-smoke-test`。
5. **训练 R1B。** 种子 0。
6. **补种子。** R1B 跑完后，自动提交种子 1 与种子 2。
7. **写报告。** `docs/reports/R1.md`。

### 四、训练运行

- 启动前 dry run 20 个迭代，确认损失，显存，日志，ckpt 路径。
- 用调度器模板 `scripts/slurm/train.sbatch`。没有调度器就用 tmux 加 nohup。
- 支持 `--resume auto`，从最新 ckpt 继续，被抢占后可恢复。
- 固定随机种子：python，numpy，torch，DataLoader worker。
- 日志格式见第七节。
- 每个 run 写 `run_manifest.json`，见 skill `bevflow-workspace-layout`。

### 五、ckpt 保存

项目里已知静态指标对 ckpt 很敏感，后面要做"最后 10 个 ckpt 的均值与标准差"。所以保存最后 10 个 epoch 的 ckpt，另存 `last.pt`。报告同时给最后一个 epoch 的数，以及最后 10 个的均值与标准差。要挑 ckpt，用验证集，并在配置里提前写好规则。

### 六、参考值对照

下表是 MotionNet 在三篇论文里的自报值，单位米，转引自项目文档，未逐篇核对原文。

| 来源 | static | slow | fast |
|---|---|---|---|
| WeakMotion 表 | 0.0201 | 0.2292 | 0.9454 |
| LEGO-Motion 表 | 0.0239 | 0.2467 | 1.0109 |
| SelfMotion 表 | 0.0256 | 0.2565 | 1.0744 |

三篇之间 static 的跨度约 27%，fast 约 14%，来自协议实现的差异。所以看包络，不看单个数：每档取三行的最小与最大。

官方权重评测值，以及 R1A 与 R1B 的结果，落在包络附近就算对得上，宽松一点没关系。对不上时，按这个顺序查：split，数据卡参数，评测阈值，自车补偿，背景权重，归一化。排查结论写进报告。这是对照，不是必须通过的门，对不上也继续推进后面的工作。

### 七、日志与结果

#### 日志布局

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

#### 结果追溯

- `metrics.json` 的 `meta` 含：run_id，ckpt 路径，git 哈希，配置哈希，数据卡哈希。
- 工作区有未提交改动时，`run_manifest.json` 记 `git_dirty: true`，并把 `git diff` 存成 run 目录下的 `code.diff`。
- 结果表由 `scripts/make_table.py` 生成。输入是 run_id 列表，输出是 markdown。文件头写生成命令，git 哈希，run_id 列表。
- 表里放上面三行参考值，注明"转引，未核对"。
- 报告里的每个数字，都能顺着"表，run_id，metrics.json"找到来源。结论句指向表，不直接写数。

#### registry

- `$RESULTS/registry.jsonl` 是唯一的真源。一行一个 run：run_id，exp_id，配置，种子，状态，三档 EPE，实验区路径，结果区路径，git 哈希，起止时间。
- 状态取 `running`，`done`，`failed`。只追加，不改旧行。同一个 run_id 以最后一行为准。
- 由 `scripts/registry.py` 在训练开始，训练结束，评测结束时写入。
- `docs/experiments/registry.md` 由 `scripts/registry.py --md` 从 jsonl 生成，文件头写"自动生成，不手改"。

---

<!-- FILE: skills/bevflow-library/SKILL.md -->

## 文件：`skills/bevflow-library/SKILL.md`

---
name: bevflow-library
description: 把 BEVflow 整理成可开源的算法库时使用，涵盖 PriorMotion，LEGO-Motion 与后续新方法。覆盖库的定位，接口约定，包布局，model zoo，许可与出处审计，私有信息清理，测试，文档，发布准备。
---

### 目标

最终要得到一个算法库：MotionNet，PriorMotion，LEGO-Motion，以及后面的新方法，共用一套数据，训练，评测。别人 `uv pip install` 之后，一条命令能复现论文里的数字。

库的名字暂用 `bevflow`。要改，问我。

### 一、接口约定

接口是库的骨架。先设计，我批准后冻结，再写模型。内容写进 `docs/design/interfaces.md`，由 agent 读完上游与旧代码后提出草案。

需要定下来的几件事：

1. **batch**：是一个 dict。列出每个键，形状，dtype，坐标约定。
2. **模型输出**：是一个 dict。列出必有的键与可选的键。额外信息放进 `aux`。
3. **模型方法**：
   - `forward(batch)` 返回输出 dict。
   - `predict(batch)` 返回确定性的预测，评测器只用它。生成式模型返回期望流。
   - `sample(batch, k)` 可选，生成式模型实现，返回 k 个样本。
4. **损失**：输入输出 dict 与 batch，返回各分项损失的 dict，以及总损失。
5. **指标**：输入预测与 GT，返回数值 dict。

命名统一用 snake_case，形状在接口文档里写明。MotionNet 上游的命名与我们的命名不同时，在 `models/compat.py` 做映射，保证官方权重能加载。

默认值标【默认】，我可以改。

### 二、包布局

```
bevflow/
├── preprocess/
├── datasets/
├── models/
│   ├── common/          共享部件：主干，头，损失，工具
│   ├── motionnet/
│   ├── priormotion/
│   ├── legomotion/
│   └── registry.py
├── trainers/
├── evaluators/
└── utils/
```

规则：

- 一个模型一个子包。子包之间不互相 import 内部细节，共用的东西下沉到 `common`。
- 新增一个模型，不改训练器与评测器。
- 配置里写模型名，registry 按名字构建。

### 三、model zoo

每个条目是一个已训练的 run，附这些东西：

| 项 | 来源 |
|---|---|
| 配置 | `configs/exp/<paper>_official.yaml` |
| 权重 | 对应 run 的 ckpt，注明 run_id |
| 指标 | `metrics.json` |
| 参数量与延迟 | 评测脚本实测 |
| 训练日志摘要 | `train.jsonl` 的关键曲线数据 |

`docs/model_zoo.md` 由脚本从 `registry.jsonl` 生成，不手写。

### 四、许可与出处审计

这是开源前最容易出事的地方。agent 只整理事实，不下法律结论，不替我选许可证。

1. 写 `docs/design/provenance.md`，逐目录逐文件标注来源：我自己写的，MotionNet 派生，第三方。
2. 已知的约束，来自项目笔记，需核对：
   - MERL 版 MotionNet 代码是 AGPL-3.0-or-later，其中自带的 nuScenes-devkit 文件是 CC-BY-NC-SA-4.0。
   - 凡是从 MotionNet 复制或改写的代码，库的许可要与之兼容。
   - nuScenes 数据的许可，项目笔记写的是未核实，社区通行 CC BY-NC-SA 4.0。用它训练的权重，发布前要核对条款。
3. 列出可选方案，交给我决定。例如：整个库采用兼容的许可证。或者把 MotionNet 派生部分隔离成可选子包。或者对共享部分从论文重写。
4. 保留所有原始版权头与许可文件，不删。
5. 在我决定之前，不公开仓库，不上传权重。

### 五、私有信息清理

写 `scripts/audit_release.sh`，在发布前扫描：

- 绝对路径：`/home/kaneqian`，`/media/nas_users`。
- 用户名，邮箱，内部主机名。
- 疑似 token 或密钥。
- 超过 1MB 的文件。

配置里的路径一律经过 `paths.py`。README 与 docs 里不出现集群路径。

### 六、测试

- 单元测试能在 CPU 上、不依赖 nuScenes 跑通：随机张量过一遍每个模型的 forward，loss，predict。
- 依赖数据的测试用 pytest marker 标出，默认不跑。
- 每个模型至少一个单 batch 过拟合测试，数据可用合成的。

### 七、文档与打包

- `README.md`：安装（`uv pip install -e .`），数据准备，快速开始，model zoo 表，引用的 BibTeX。
- `docs/` 沿用现有分类。对外的内容和内部过程文档分开：`ref/`，`devlog` 一类不发布。
- `pyproject.toml` 声明依赖。用 `uv build` 验证能打包。
- 写 `CHANGELOG.md`。第一个版本标 `v0.1`。

### 八、发布前清单

1. 接口文档已冻结，所有模型符合。
2. 出处审计完成，我已决定许可方案。
3. `audit_release.sh` 无告警。
4. CPU 单元测试通过。
5. model zoo 里每个条目的指标可从 registry 复现。
6. README 里的快速开始，在干净环境里走通一遍。

发布本身，包括推送公共仓库与上传权重，由我来做。

---

<!-- FILE: skills/bevflow-refactor/SKILL.md -->

## 文件：`skills/bevflow-refactor/SKILL.md`

---
name: bevflow-refactor
description: 冒烟通过后，把 MotionNet 上游代码重构为 BEVflow 的模块化结构时使用。覆盖五个模块的划分，配置体系，上游文件映射，保证行为不变的测试，以及给后续新方法预留的最小接口。
---

### 目标

只改结构，不改行为。重构之后，新方法只需要改一个模块。

### 前提

`smoke.sh` 在上游代码上已经通过。没有这个基线，不开始。

### 目标结构

```
bevflow/
├── preprocess/    数据生成，体素化，划分，数据卡读写
├── datasets/      Dataset 与 DataLoader
├── models/        网络，损失，head，registry
├── trainers/      训练循环，优化器与调度器，ckpt，resume
├── evaluators/    指标，评测循环，可视化
└── utils/         paths，seed，config，registry
configs/
├── base.yaml
├── data/
├── model/
└── exp/           每个实验一个文件，用 _base_ 继承
scripts/           train.py  eval.py  viz.py  gen_data.py  薄入口
docs/              setup  design  experiments  reports  reference
```

### 做法

#### 1. 先读后动

读完上游全部相关脚本，写 `docs/design/upstream_map.md`：上游文件与函数，对应的新位置。预期映射：

| 上游内容 | 新位置 |
|---|---|
| 数据生成脚本 | `preprocess/` |
| 数据读取与 DataLoader | `datasets/` |
| 模型定义 | `models/` |
| 损失函数 | `models/losses.py` |
| 训练脚本 | `trainers/` |
| 位移评测与分类评测 | `evaluators/` |

#### 2. 一个模块一个模块搬

搬完就跑对应测试，通过后提交。顺序：`utils`，`preprocess`，`datasets`，`models`，`evaluators`，`trainers`，最后是 `scripts` 与 `configs`。

#### 3. 保留 state_dict 键名

参数名字和层级与上游一致，官方权重才能直接加载。确实要改名，在 `models/compat.py` 写显式映射。

#### 4. 配置

YAML 加简单的命令行覆盖，形如 `--set optim.lr=1e-3`。不引入重型框架。每个实验一个 `configs/exp/<exp_id>.yaml`，用 `_base_` 继承。训练开始时把解析后的完整配置写进 run 目录的 `config_snapshot.yaml`。

#### 5. 最小的扩展接口

只做这几个：

- `models` 的 registry：按名字构建 backbone 与 head，后面换解码头只改配置。
- `losses` 的 registry。
- `evaluators` 的 metric registry：加新指标只加一个函数。

不为将来可能的需要写没用到的抽象。

### 行为不变的测试

重构前，用上游代码生成一组小的 golden 数据，放 `tests/golden/`。重构后比对：

| 测试 | 内容 | 容差 |
|---|---|---|
| 预处理 | mini 上新旧输出数组 | 完全相等 |
| 前向 | 加载官方权重，同一输入，输出 | `allclose`，atol 1e-5 |
| 损失与指标 | 同一份预测与 GT | `allclose` |

短训练的损失曲线对比是可选项。不一致时先查数据顺序与随机数，再查代码。

### 检查点

1. 上面的测试通过。
2. 在新结构上重跑 `smoke.sh` 通过。
3. `third_party/MotionNet` 没被改动。
4. 文档按分类放好。

### 不做

- 不改默认超参。
- 不顺手优化算法。
- 不改张量含义和坐标约定。

---

<!-- FILE: skills/bevflow-reverse-reproduce/SKILL.md -->

## 文件：`skills/bevflow-reverse-reproduce/SKILL.md`

---
name: bevflow-reverse-reproduce
description: 用户触发第六阶段后使用。以论文 PDF 为规格，以旧代码为参考实现，把 PriorMotion 与 LEGO-Motion 逆向复现到 BEVflow 并训练出基线。覆盖材料放置，规格提取，旧代码审计，决策清单，数据层先行，模型移植，训练对照，预留接口。
---

### 触发与材料

用户说"开始第六阶段"才执行。材料放在 `$CODE/ref/<name>/`，`<name>` 取 `priormotion` 或 `legomotion`：

```
ref/<name>/
├── paper.pdf
├── code/        旧代码，只读
└── notes.md     用户的备注：已知问题，代码对应的版本，哪些部分不可信
```

`ref/` 写进 `.gitignore`，不进仓库，也不参与开源。缺什么，问我：旧项目有没有训练日志，中间 ckpt，论文以外的实验记录。有 ckpt 是最强的对照。

### 原则

1. 规格以论文为准。旧代码是参考实现，不是真值。
2. 旧代码的结构，数据加载，评测都可能有错。不继承，要审计。
3. 论文和旧代码冲突，不静默选择。写进决策清单，等我裁决。
4. 目标是在统一管线上训练出论文表里的数字，不是逐比特复刻旧代码。
5. 目标是算法库。共享部件抽到 `models/common/`，每个模型是一个能独立 import 的子包。
6. 每个结论都能追溯：论文的页码，公式号，表号，或旧代码的文件与行号。

### 基线现状

- PriorMotion 与 LEGO-Motion 没有现成权重，靠复现再训练。
- BE-STI 没有官方代码，只引论文数字。
- MotionNet 的管线与权重已在 P4 打通，作为共同的底座。

### 步骤

#### 1. 规格提取，产出 `docs/design/spec_<name>.md`

读 PDF 的办法：

- 若装有 `pdf-reading` skill，先加载它。
- 没有就用 `pdftotext -layout` 取正文，用 `pdfplumber` 取表格。
- 公式在文本里常常乱码。网络通且有 arXiv 版时，可取 arXiv 的 LaTeX 源码核对公式。取不到就把乱码处标【待确认】，问我。
- 不用 Read 读图片。架构图看不了，就读图注和正文描述，并请我口述或贴图。

规格文档的内容，每条标注页码，公式号或表号：

| 项 | 内容 |
|---|---|
| 任务设定 | 输入帧数，坐标系，输出含义，预测时长 |
| 网络 | 逐模块：输入输出张量形状，结构，公式 |
| 损失 | 每一项的公式，权重，调度 |
| 训练配方 | epoch，学习率，优化器，batch size，增广，硬件 |
| 评测 | 指标定义，速度分档，评测集 |
| 报告数字 | 论文里的每张表，每个数 |
| 消融 | 每个消融的设置与结果，后面当作验证目标 |
| 效率 | 参数量，延迟及分解 |

读不懂或前后矛盾的地方，单列"歧义清单"。

#### 2. 旧代码审计，产出 `docs/design/audit_<name>.md`

建一张追溯表：

| 规格条目 | 旧代码位置 | 状态 | 说明 |
|---|---|---|---|

状态取一致，不一致，缺失，多余，不确定。不修改旧代码。

数据层单独做一份专项检查，每项都要给出结论：

- 输入帧数与时间对齐。
- 自车补偿与坐标系。
- 体素化的范围，分辨率，高度 bin。
- 标签生成：位移真值，动静划分，类别。
- 额外的数据产物：先验编码用的运动场真值，实例标签，向量化数据。
- 划分：是否与 500/100/250 一致，有无泄漏。
- 增广：种类，是否只用于训练。
- 评测代码：三档阈值，分类指标，与 P4 的评测器有何不同。

#### 3. 决策清单，产出 `docs/design/decisions_<name>.md`

把所有"不一致"与"歧义"整理成编号问题，一次交给我批量裁决：

```
D1 <一句话问题>
 现象：
 证据：论文 p.x 式(y)，旧代码 file:line
 选项：A ... B ...
 建议：
```

我回复后，把结果记进同一个文件，标【用户确认】。之后的实现按决策走。等待回复时，继续做不依赖答案的部分。

#### 4. 库设计

先设计，再写码。设计内容见 skill `bevflow-library`：接口约定，目录规划，共享部件清单。设计文档我批准后冻结。

#### 5. 数据层先行

先做数据，再做模型。数据错了，后面全是白费。

- 与 MotionNet 共享的输入，直接复用 `motionnet_v1`。额外的产物，生成到新版本目录 `$DATA/preprocessed/<data_version>_<name>`，写新的数据卡。
- 数据层测试：
  - 共享输入与 `motionnet_v1` 逐样本一致。
  - 标签的速度三档占比与 `data_stats.json` 一致。
  - 自车补偿检查：GT 静态区域的位移接近零。
  - 抽 20 个样本做形状，dtype，NaN 检查。
- 旧加载器的输出当对照。与新实现不一致时，查明谁错了，对照论文与 MotionNet，结论记进审计文档。

#### 6. 模型移植

顺序：共享主干，输出头，损失，训练调度。一次一个模块，每个模块做完再做下一个。

- 形状与数值合理性：每个模块都做。
- 与旧代码对比：只在旧代码可运行，且审计认为该模块与论文一致时才做。同权重，同输入，对比输出。
- 不一致的模块，以论文为准重写，写明原因。

#### 7. 训练与对照

1. 短训练：单 batch 过拟合，确认各分支都有梯度，各损失项都在下降。
2. 完整训练：按论文训练配方。论文没写的，用项目预算：45 epoch，单卡 bs4，或 bs2 加梯度累积 2。保存最后 10 个 epoch 的 ckpt。流程和日志规范沿用 skill `bevflow-experiment-run`。
3. 评测并对照：

| 对照项 | 参考值 |
|---|---|
| 论文报告数字 | spec 文档里的表 |
| 参数量与延迟 | MotionNet 7.9M，19 ms。LEGO-Motion STPN 8.1M，21 ms。PriorMotion 11.5M，69 ms，其中体素化 12 ms，前向 57 ms。来自项目笔记 |
| 消融趋势 | 论文消融里每个组件的方向，在我们的实现里是否一致 |

参数量应当基本一致，延迟同量级。EPE 对不上时，按这个顺序查：数据，损失权重与调度，模块。每次尝试写进 `docs/reports/reproduce_<name>.md` 的排查记录。论文里写明的训练配方照着用，不为了对齐数字而调参。

4. 旧项目里若有 ckpt，直接加载到新模型对比输出。有就做，没有就靠训练复现。

#### 8. 为后续实验预留的接口

**PriorMotion 侧**

- `latent_source` 取 `student` 或 `teacher`。teacher 用真值先验编码。用来做置换测试。
- 一次前向返回 K 个样本。用来算 best-of-K 与 K 样本方差。
- 评测器按速度分箱输出。

**评测器侧**

- 可选开关：GT 静态掩码做完美门控，去掉后处理，扫描背景权重。
- 近邻 ckpt 权重平均的评测入口。
- 指标注册位：static P95，K 样本方差，best-of-K。

### 并行

两个模型共享数据层与主干。先把 `common` 部分做完并合入主分支，再各开一个 git worktree 分支，并行做 `priormotion` 与 `legomotion`。合并前跑全部测试。

### 产物

- `docs/design/spec_<name>.md`，`audit_<name>.md`，`decisions_<name>.md`
- `bevflow/models/<name>/`，共享部分在 `bevflow/models/common/`
- `docs/reports/reproduce_<name>.md`：对照表，偏差，排查记录

### 不做

- 复现的同时不做方法改进。
- 不改 `ref/` 里的任何东西。

---

<!-- FILE: skills/bevflow-smoke-test/SKILL.md -->

## 文件：`skills/bevflow-smoke-test/SKILL.md`

---
name: bevflow-smoke-test
description: 在 nuScenes mini 上用小批量验证 BEVflow 全链路时使用。覆盖数据生产，训练，ckpt 存取，评测，可视化五步，显存与耗时实测。重构前后都要重跑。
---

### 目标

用最小代价证明数据，训练，ckpt，评测，可视化能连通。写成一个脚本，重构前后各跑一次。

### 脚本

写 `scripts/smoke.sh`。它依次跑五步，打印一张 PASS 或 FAIL 表，产物放 `$RESULTS/_smoke/<timestamp>/`。某步失败，后面标 SKIP。

数据源是 `$DATA/preprocessed/mini`。重构前它调用上游脚本，重构后调用新入口，检查项相同。

### 五步

下表的阈值是初始设定，不合理就改，改了在 `docs/setup/smoke.md` 记一行。

| 步骤 | 做什么 | 通过条件 |
|---|---|---|
| S1 数据 | mini 上生成 train 与 val，加载 20 个样本 | 形状正确，无 NaN 与 Inf |
| S2 训练 | 100 到 300 个迭代。另做单 batch 过拟合 | 损失有限且下降。单 batch 重复 200 步，损失至少降一半 |
| S3 存取 | 存 ckpt，重新加载，同一输入前向两次 | 输出最大绝对差小于 1e-6 |
| S4 评测 | 评测器在 mini val 上评测 S2 的 ckpt | 生成 `metrics.json`，数值有限，字段齐全 |
| S5 可视化 | 固定 4 个样本，出 GT，预测，误差图 | 图片文件通过数值检查 |

单 batch 过拟合是最便宜的训练正确性检查。它不过，先查数据与损失，再查优化器。

### metrics.json 字段

```json
{
  "run_id": "", "ckpt": "", "split": "val", "n_samples": 0,
  "epe": {"static": {"mean": 0, "median": 0},
          "slow":   {"mean": 0, "median": 0},
          "fast":   {"mean": 0, "median": 0}},
  "cls": {"OA": 0, "MCA": 0, "per_class": {}},
  "meta": {"git": "", "config_sha": "", "data_card_sha": "", "eval_protocol": "motionnet_v1"}
}
```

三档按上游定义：静态，不超过 5 m/s，超过 5 m/s。字段名固定，后面所有实验沿用。

### 可视化

内容：BEV 网格，颜色表示流的方向与大小，色轮与色标固定。每个样本画 GT，预测，误差三联图。另画一张静态误差图：只画 GT 为静态的格子，颜色表示预测流的模，后面用来看静态噪点。

样本固定在 `configs/viz_samples.yaml`：val 里选 4 个，分别是静态为主，有慢速目标，有快速目标，固定种子随机抽一个。

验证图片用数值检查，写成 `scripts/check_viz.py`：文件存在且大于 5KB，宽高正确，像素标准差不为零，预测图与 GT 图不逐像素相同。再写 `viz_manifest.json`，汇报时告诉我图的目录，我自己打开看。

不用 Read 读 PNG。部分模型不支持工具结果里的图片，读一次会让会话损坏。遇到 `400 unsupported content block`，停下，让我开新会话。

### 资源实测

S2 里顺带测，写进 `docs/setup/resources.md`：

- bs4 与 bs2 的显存峰值，用 `torch.cuda.max_memory_allocated`。
- 单步耗时，数据加载耗时占比。
- 据此估算 45 个 epoch 的总时长。
- bs4 放不下，就用 bs2 加梯度累积 2，等效 batch 不变，记录这个决定。

### 排错

按这个顺序查：数据形状，数据值域，损失定义，学习率，优化器。每一步先打印再猜。判断不了就问我。

---

<!-- FILE: skills/bevflow-workspace-layout/SKILL.md -->

## 文件：`skills/bevflow-workspace-layout/SKILL.md`

---
name: bevflow-workspace-layout
description: 在集群上创建、检查或使用 BEVflow 的目录与路径时使用。覆盖代码区与实验区的分工，项目二级目录隔离，软链接，路径配置，run 命名与产物清单。
---

### 目标

代码区只放代码和简要 docs，大文件全部在 NAS。本项目只占用每个根下名为 `bevflow` 的二级目录。

### 变量

沿用总控提示词第 0 节的 `CODE`，`NAS`，`EXP`，`RESULTS`，`DATA`，`CACHE`。

### 目录树

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

### 创建

1. 对每个 NAS 根，`ls` 看二级目录名，只确认 `bevflow` 在不在，不进入别的项目。
2. `mkdir -p` 建树。已存在就复用。
3. 在 `$CODE` 里建三个软链接，用绝对路径，用 `readlink -f` 确认。把软链接和 `ref/` 写进 `.gitignore`。

### 路径来源只有一个

代码里不写绝对路径。所有路径经过 `bevflow/utils/paths.py`。它读环境变量 `BEVFLOW_CODE`，`BEVFLOW_EXP`，`BEVFLOW_RESULTS`，`BEVFLOW_DATA`，缺省值取自 `configs/paths.yaml`。

写文件的入口先 `assert` 目标路径的 `realpath` 在这四个根之下。

### run 命名与清单

- `exp_id` 沿用实验卡编号，如 `R1A`。
- `run_id` 为 `YYYYMMDD_<exp_id>_<model>_s<seed>`。
- 每个 run 写 `run_manifest.json`：git 哈希，是否有未提交改动，配置哈希，数据卡哈希，种子，主机，GPU，起止时间，启动命令。
- `registry.jsonl` 由脚本写入，规则见 skill `bevflow-experiment-run` 第七节。

### NAS 小文件

NAS 读小文件慢时，把当前 job 要用的数据复制到本地盘训练，训练完丢弃。永久位置仍然是 `$DATA`。
