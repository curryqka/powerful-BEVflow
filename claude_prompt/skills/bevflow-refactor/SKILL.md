---
name: bevflow-refactor
description: 冒烟通过后，把 MotionNet 上游代码重构为 BEVflow 的模块化结构时使用。覆盖五个模块的划分，配置体系，上游文件映射，保证行为不变的测试，以及给后续新方法预留的最小接口。
---

# 目标

只改结构，不改行为。重构之后，新方法只需要改一个模块。

# 前提

`smoke.sh` 在上游代码上已经通过。没有这个基线，不开始。

# 目标结构

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

# 做法

## 1. 先读后动

读完上游全部相关脚本，写 `docs/design/upstream_map.md`：上游文件与函数，对应的新位置。预期映射：

| 上游内容 | 新位置 |
|---|---|
| 数据生成脚本 | `preprocess/` |
| 数据读取与 DataLoader | `datasets/` |
| 模型定义 | `models/` |
| 损失函数 | `models/losses.py` |
| 训练脚本 | `trainers/` |
| 位移评测与分类评测 | `evaluators/` |

## 2. 一个模块一个模块搬

搬完就跑对应测试，通过后提交。顺序：`utils`，`preprocess`，`datasets`，`models`，`evaluators`，`trainers`，最后是 `scripts` 与 `configs`。

## 3. 保留 state_dict 键名

参数名字和层级与上游一致，官方权重才能直接加载。确实要改名，在 `models/compat.py` 写显式映射。

## 4. 配置

YAML 加简单的命令行覆盖，形如 `--set optim.lr=1e-3`。不引入重型框架。每个实验一个 `configs/exp/<exp_id>.yaml`，用 `_base_` 继承。训练开始时把解析后的完整配置写进 run 目录的 `config_snapshot.yaml`。

## 5. 最小的扩展接口

只做这几个：

- `models` 的 registry：按名字构建 backbone 与 head，后面换解码头只改配置。
- `losses` 的 registry。
- `evaluators` 的 metric registry：加新指标只加一个函数。

不为将来可能的需要写没用到的抽象。

# 行为不变的测试

重构前，用上游代码生成一组小的 golden 数据，放 `tests/golden/`。重构后比对：

| 测试 | 内容 | 容差 |
|---|---|---|
| 预处理 | mini 上新旧输出数组 | 完全相等 |
| 前向 | 加载官方权重，同一输入，输出 | `allclose`，atol 1e-5 |
| 损失与指标 | 同一份预测与 GT | `allclose` |

短训练的损失曲线对比是可选项。不一致时先查数据顺序与随机数，再查代码。

# 检查点

1. 上面的测试通过。
2. 在新结构上重跑 `smoke.sh` 通过。
3. `third_party/MotionNet` 没被改动。
4. 文档按分类放好。

# 不做

- 不改默认超参。
- 不顺手优化算法。
- 不改张量含义和坐标约定。
