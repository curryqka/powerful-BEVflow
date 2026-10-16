---
name: bevflow-smoke-test
description: 在 nuScenes mini 上用小批量验证 BEVflow 全链路时使用。覆盖数据生产，训练，ckpt 存取，评测，可视化五步，显存与耗时实测。重构前后都要重跑。
---

# 目标

用最小代价证明数据，训练，ckpt，评测，可视化能连通。写成一个脚本，重构前后各跑一次。

# 脚本

写 `scripts/smoke.sh`。它依次跑五步，打印一张 PASS 或 FAIL 表，产物放 `$RESULTS/_smoke/<timestamp>/`。某步失败，后面标 SKIP。

数据源是 `$DATA/preprocessed/mini`。重构前它调用上游脚本，重构后调用新入口，检查项相同。

# 五步

下表的阈值是初始设定，不合理就改，改了在 `docs/setup/smoke.md` 记一行。

| 步骤 | 做什么 | 通过条件 |
|---|---|---|
| S1 数据 | mini 上生成 train 与 val，加载 20 个样本 | 形状正确，无 NaN 与 Inf |
| S2 训练 | 100 到 300 个迭代。另做单 batch 过拟合 | 损失有限且下降。单 batch 重复 200 步，损失至少降一半 |
| S3 存取 | 存 ckpt，重新加载，同一输入前向两次 | 输出最大绝对差小于 1e-6 |
| S4 评测 | 评测器在 mini val 上评测 S2 的 ckpt | 生成 `metrics.json`，数值有限，字段齐全 |
| S5 可视化 | 固定 4 个样本，出 GT，预测，误差图 | 图片文件通过数值检查 |

单 batch 过拟合是最便宜的训练正确性检查。它不过，先查数据与损失，再查优化器。

# metrics.json 字段

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

# 可视化

内容：BEV 网格，颜色表示流的方向与大小，色轮与色标固定。每个样本画 GT，预测，误差三联图。另画一张静态误差图：只画 GT 为静态的格子，颜色表示预测流的模，后面用来看静态噪点。

样本固定在 `configs/viz_samples.yaml`：val 里选 4 个，分别是静态为主，有慢速目标，有快速目标，固定种子随机抽一个。

验证图片用数值检查，写成 `scripts/check_viz.py`：文件存在且大于 5KB，宽高正确，像素标准差不为零，预测图与 GT 图不逐像素相同。再写 `viz_manifest.json`，汇报时告诉我图的目录，我自己打开看。

不用 Read 读 PNG。部分模型不支持工具结果里的图片，读一次会让会话损坏。遇到 `400 unsupported content block`，停下，让我开新会话。

# 资源实测

S2 里顺带测，写进 `docs/setup/resources.md`：

- bs4 与 bs2 的显存峰值，用 `torch.cuda.max_memory_allocated`。
- 单步耗时，数据加载耗时占比。
- 据此估算 45 个 epoch 的总时长。
- bs4 放不下，就用 bs2 加梯度累积 2，等效 batch 不变，记录这个决定。

# 排错

按这个顺序查：数据形状，数据值域，损失定义，学习率，优化器。每一步先打印再猜。判断不了就问我。
