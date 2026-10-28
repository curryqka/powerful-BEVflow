# LEGO-Motion 接口设计与包布局（C4 草案）

依据：spec_legomotion.md（论文为规格）+ ref/legomotion/notes.md（用户记忆）。遵循 skill `bevflow-library`：
batch 是 dict、模型输出是 dict、`forward`/`predict`/`sample` 约定、共享部件下沉 `common`。

## 1. 接口约定（lea 与 PriorMotion 共享同一套）

### 1.1 batch（dict）

| 键 | 形状 | dtype | 说明 |
|---|---|---|---|
| `voxel` | (B, 5, 256, 256, 13) | float32 | 与 motionnet_v1 完全一致 |
| `pixel_cat` | (B, 256, 256, 5) | float32 | one-hot 或 soft |
| `disp_gt` | (B, 20, 256, 256, 2) | float32 | 未来 20 帧位移（米） |
| `valid` | (B, 20, 256, 256) | float32 | 有效像素 |
| `non_empty` | (B, 256, 256) | float32 | 占据掩码 |
| `motion_gt` | (B, 256, 256, 2) | float32 | 动静 one-hot |
| `instance_map` | (B, 256, 256) | uint8 | 实例 id（LEGO 掩码监督 + 稳定性指标） |
| `trans_matrices` | (B, 20, 4, 4) | float32 | 一致性损失用（可选） |

（训练集现有 10 元组直接映射为 dict；评测集需把 DatasetSingleSeq 的 return_instance 打开。）

### 1.2 模型输出（dict）

| 键 | 形状 | 说明 |
|---|---|---|
| `disp` | (B×20, 2, 256, 256) | 相对逐帧位移，与现有评测器兼容 |
| `cell_logits` | (B, 5, 256, 256) | |
| `motion_logits` | (B, 2, 256, 256) | |
| `pred_masks` | (B, N, 256, 256) | 实例二值掩码 logits（sigmoid 前） |
| `pred_classes` | (B, N, Kc+1) | 掩码类别 logits，Kc+1 含 no-object |
| `aux` | dict | 额外信息（如 vae_dict / 匹配指示） |

### 1.3 方法

- `forward(batch)` → 输出 dict（训练用）。
- `predict(batch)` → 只含确定性预测的 dict（评测器只用它）。
- `sample(batch, k)` → 仅生成式模型（PriorMotion）实现；LEGO 不需要。

### 1.4 损失接口

`legomotion_loss(outputs, batch, lambda_schedule) -> (total, parts_dict)`：
- 网格三项直接复用 `models/losses.py` 的加权 SmoothL1/CE（同 MotionNet 权重 0.005/1.0）。
- 掩码项在此模块内实现：匈牙利匹配 + `5.0·BCE + 5.0·Dice`，匹配代价 `1.0·(−p_cls) + 5.0·BCE + 5.0·Dice`，no-object 0.1 降权。
- 两阶段调度（λ：前 20 epoch mask=2.0；后 25 epoch 网格项=2.0）由 trainer 注入。

## 2. 包布局（P7 目标，现逐步靠拢）

```
bevflow/models/
├── common/           # 从 motionnet.py/priormotion.py 下沉：STPN、CellClass/StateEst、Conv3D
├── motionnet/        # motionnet.py（现有）
├── priormotion/      # priormotion.py + prior_extractor.py + vae（现有，P7 时整理为子包）
├── legomotion/
│   ├── __init__.py
│   ├── model.py      # BEV Encoder + IaIE + IeBE + 头
│   ├── iaie.py       # 实例交互编码器（式1-2）
│   ├── iebe.py       # 早期/晚期融合（式3-6）
│   └── losses.py     # 匹配 + 掩码损失
└── registry.py       # 按名构建：motionnet / priormotion / priormotion_rvpe / legomotion
```

## 3. 待你批准确认的 6 项歧义（spec 末节）

| # | 问题 | 我的建议 |
|---|---|---|
| 1 | 实例 query N 与 D | N=256，D=64（你的记忆） |
| 2 | IaIE 自注意力层数 | 1 层 self-attn + 1 层 cross-attn（Pre-LN，残差）；D=64 |
| 3 | IeBE 晚期融合 FFN 维度 | 保持 D=64 两层 FFN |
| 4 | 掩码头 BEV 特征用哪一层 | IeBE 晚期输出 E（最丰富） |
| 5 | 掩码类别数 Kc | 5（bg+车+人+车行+其他）；no-object=第 6 类 |
| 6 | 训练预算 | 论文口径 45ep/bs4/Adam1.6e-3（与 R1B 一致），两阶段 λ 调度照抄 |

## 4. C5 数据层测试计划（无需新数据）

motionnet_v1 已含 instance_map；测试项：
1. batch 字段形状/dtype/finite（抽 20 样本）。
2. 实例图与 pixel_cat 一致性（每个实例 id 内类别占优比例）。
3. 与 motionnet_v1 的逐样本一致（已由既有测试保证，复用）。
4. 训练集 10 元组 → dict 适配器单测（trainer 专用）。

设计写到这里，等你批复第 3 节的 6 项后冻结，然后进入 C6 实现。
