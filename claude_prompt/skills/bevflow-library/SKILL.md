---
name: bevflow-library
description: 把 BEVflow 整理成可开源的算法库时使用，涵盖 PriorMotion，LEGO-Motion 与后续新方法。覆盖库的定位，接口约定，包布局，model zoo，许可与出处审计，私有信息清理，测试，文档，发布准备。
---

# 目标

最终要得到一个算法库：MotionNet，PriorMotion，LEGO-Motion，以及后面的新方法，共用一套数据，训练，评测。别人 `uv pip install` 之后，一条命令能复现论文里的数字。

库的名字暂用 `bevflow`。要改，问我。

# 一、接口约定

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

# 二、包布局

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

# 三、model zoo

每个条目是一个已训练的 run，附这些东西：

| 项 | 来源 |
|---|---|
| 配置 | `configs/exp/<paper>_official.yaml` |
| 权重 | 对应 run 的 ckpt，注明 run_id |
| 指标 | `metrics.json` |
| 参数量与延迟 | 评测脚本实测 |
| 训练日志摘要 | `train.jsonl` 的关键曲线数据 |

`docs/model_zoo.md` 由脚本从 `registry.jsonl` 生成，不手写。

# 四、许可与出处审计

这是开源前最容易出事的地方。agent 只整理事实，不下法律结论，不替我选许可证。

1. 写 `docs/design/provenance.md`，逐目录逐文件标注来源：我自己写的，MotionNet 派生，第三方。
2. 已知的约束，来自项目笔记，需核对：
   - MERL 版 MotionNet 代码是 AGPL-3.0-or-later，其中自带的 nuScenes-devkit 文件是 CC-BY-NC-SA-4.0。
   - 凡是从 MotionNet 复制或改写的代码，库的许可要与之兼容。
   - nuScenes 数据的许可，项目笔记写的是未核实，社区通行 CC BY-NC-SA 4.0。用它训练的权重，发布前要核对条款。
3. 列出可选方案，交给我决定。例如：整个库采用兼容的许可证。或者把 MotionNet 派生部分隔离成可选子包。或者对共享部分从论文重写。
4. 保留所有原始版权头与许可文件，不删。
5. 在我决定之前，不公开仓库，不上传权重。

# 五、私有信息清理

写 `scripts/audit_release.sh`，在发布前扫描：

- 绝对路径：`/home/kaneqian`，`/media/nas_users`。
- 用户名，邮箱，内部主机名。
- 疑似 token 或密钥。
- 超过 1MB 的文件。

配置里的路径一律经过 `paths.py`。README 与 docs 里不出现集群路径。

# 六、测试

- 单元测试能在 CPU 上、不依赖 nuScenes 跑通：随机张量过一遍每个模型的 forward，loss，predict。
- 依赖数据的测试用 pytest marker 标出，默认不跑。
- 每个模型至少一个单 batch 过拟合测试，数据可用合成的。

# 七、文档与打包

- `README.md`：安装（`uv pip install -e .`），数据准备，快速开始，model zoo 表，引用的 BibTeX。
- `docs/` 沿用现有分类。对外的内容和内部过程文档分开：`ref/`，`devlog` 一类不发布。
- `pyproject.toml` 声明依赖。用 `uv build` 验证能打包。
- 写 `CHANGELOG.md`。第一个版本标 `v0.1`。

# 八、发布前清单

1. 接口文档已冻结，所有模型符合。
2. 出处审计完成，我已决定许可方案。
3. `audit_release.sh` 无告警。
4. CPU 单元测试通过。
5. model zoo 里每个条目的指标可从 registry 复现。
6. README 里的快速开始，在干净环境里走通一遍。

发布本身，包括推送公共仓库与上传权重，由我来做。
