# 参考值（主要来源已核实）

来源 A：PriorMotion (Qian et al., ICCV 2025) Table 2/3/5/6/7——用户自己的论文，为主要来源。
此前"转引 WeakMotion/LEGO-Motion 表"的各行已溯源至此（MotionNet 在不同论文/配置下的自报值）。

## MotionNet 基线在各家论文中的自报值（test，米）

| 来源 | static | slow | fast | 备注 |
|---|---|---|---|---|
| PriorMotion Table 2, MotionNet 原版 | 0.0262 | 0.2467 | 0.9878 | **官方权重 model.pth 对应此行**（非 MGDA） |
| PriorMotion Table 2, MotionNet†(MGDA) | 0.0201 | 0.2292 | 0.9454 | MGDA 训练，官方权重不对应此行 |
| PriorMotion Table 6(a), MotionNet 基线 | 0.0240 | 0.2467 | 1.0109 | 8.0M，19ms |
| SelfMotion 表（转引） | 0.0256 | 0.2565 | 1.0744 | |

## 官方权重对照结论（2026-10-17 实测，test bs1）

| 口径 | 官方权重实测 | 论文 MotionNet 原版 | 判定 |
|---|---|---|---|
| slow@1.0s | 0.247 | 0.2467 | 几乎完全一致 → **文献口径 = 1.0s 单点值** |
| fast@1.0s | 1.011 | 0.9878 | +2.4%，可接受 |
| static（均值口径） | 0.0145 | 0.0262 | 均值 vs 单点口径差异，见 R1.md 第 4 节 |

## P6 复现目标（PriorMotion，STPN 主干）

| 指标 | 论文值 | 出处 |
|---|---|---|
| 三档 EPE（test） | 0.0251 / 0.2247 / 0.8324 | Table 2 "STPN /w(Ours)"（Table 6d 为 0.8318，两表微差，以 Table 2 为准） |
| 参数量 / 推理延迟 | 11.5M / 69ms（12ms 体素化 + 57ms 前向，RTX 3090） | Table 6(d) |
| 分类 MCA / OA | 76.3 / 93.6（OA 低于 MotionNet 96.3 属预期：生成式以分类换运动） | Table 3 |
| 逐类 | Bg 94.6 / Vehicle 92.1 / Ped 86.9 / Bike 27.3 / Others 80.4 | Table 3 |
| 长距离(>20m) | static 0.0263 / slow 0.2207 / fast 0.8549 | Table 5 |
| 运动稳定性 | 0.0088 | Table 5 |
| 泛化 GI | 86.4%（mask "other" 类） | Table 4 |
| 消融（d）全配置 | 0.0251 / 0.2247 / 0.8324，median 0/0.0949/0.6069 | Table 7(f) |

论文实现细节（§4.1）：与 MotionNet 同预处理；Adam 1.6e-3，45 epoch，bs4，单卡 A100；衰减 0.5 @ epochs 10/20/30/40——**与本项目 R1B 预算完全一致**。§4.2 声明 PriorMotion 不用 MGDA 与数据增强。

λ_mot/λ_state/λ_cls/λ_prior 的数值与 FSD/SE、SGRU 维度、动态/静态掩码来源等细节在附录/代码中——以旧代码为准（盘点进行中）。

## LEGO-Motion（P6 第二目标，代码已失，按论文+记忆重构）

用户记忆的要点：实例 query N=256，D=64；BEV 下采样 16×16 作 IaIE 的 key/value；参数增量约 0.2M；IaIE/IeBE 为 Pre-LN Transformer block（残差，晚期 cross-attention Eq. 6）；BEV 侧可学习 H×W×D 位置编码，query 侧 DETR 式可学习 embedding；L_mask = 5.0·BCE + 5.0·Dice；匹配代价 C = 1.0·(−p_cls) + 5.0·BCE + 5.0·Dice，no-object 0.1 降权；fast 差距解释 = 体素化流程与两阶段权重调度的小差异。论文 PDF 在旧代码目录内，P6 第二阶段精读。
