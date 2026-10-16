---
name: bevflow-data-pipeline
description: 处理 BEVflow 的数据时使用。覆盖原始 nuScenes 的软链接，500/100/250 划分，MotionNet 数据生产，数据卡，统计，以及 mini 冒烟数据。
---

# 目标

把 nuScenes 变成 MotionNet 协议的预处理数据，放进 `$DATA`，并能追溯到参数，划分和上游版本。

# 一、原始数据只链接

原始路径已定：`$RAW_NUSCENES`，即 `/media/nas_users/kaneqian/data/Nuscences`（总控提示词第 0 节）。

1. 确认有 `v1.0-trainval` 元数据，以及 `samples`，`sweeps`。
2. 建链接 `$DATA/raw/nuscenes -> $RAW_NUSCENES`，只读使用。
3. 路径失效时再浅层找，深度不超过 3。找不到，问我。不要全盘搜索 NAS。

# 二、划分

直接用上游 `data/split.npy`。已核实它正好是 train 500，val 100，test 250，共 850 个场景，MotionNet 协议。

复制到 `$DATA/splits/`，并 `assert` 三个数：train 500，val 100，test 250，且三者无交集。断言不过，说明上游版本不对，问我，不要自己改数。

冒烟用 mini 划分：10 个场景分成 6，2，2。

# 三、数据生产

1. 用上游的数据生成脚本。参数从上游代码读，不凭记忆。
2. 输出到 `$DATA/preprocessed/<data_version>/{train,val,test}`。正式版本叫 `motionnet_v1`，mini 版本叫 `mini`。
3. 先在 mini 上跑通并计时，估计全量耗时。耗时长，就用调度器的 array job 按场景分片。
4. 任务可恢复：已生成的样本跳过。
5. 日志写到 `$EXP/_data_logs/`。

# 四、数据卡

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

# 五、软链接到上游期望的目录

读上游训练脚本，找出它期望的数据路径。在 `$CODE` 下建对应的软链接，指向 `$DATA/preprocessed/<data_version>`。不改上游的路径默认值。

# 六、统计

写 `scripts/check_data.py`，对每个 split：

- 随机抽 20 个样本，检查形状，dtype，无 NaN 与 Inf。
- 统计 GT 速度三档占比：静态，不超过 5 m/s，超过 5 m/s。
- 统计动态格子占非空格子的比例。后面评估稀疏生成的收益要用它，现在顺手记下。

结果写进 `data_stats.json`。
