# spec_legomotion（C1 规格提取）

来源：ref/legomotion/paper.pdf（IROS 2025，8 页）。代码已失（用户确认），本 spec 为唯一规格；
用户记忆要点见 ref/legomotion/notes.md，与论文交叉核对一致。

## 任务设定

| 项 | 规格 | 出处 |
|---|---|---|
| 输入 | 5 帧体素化点云 V∈{0,1}^(H×W×C)，[-32,32]²×[-3,2]，0.25×0.25×0.4 | p.5 §IV.A |
| 输出 | 每格运动/分类/状态 + 每实例二值掩码（含类别与 state） | p.3 §III.A |
| 骨干 | STPN（可插拔 STT/STI） | p.3 §III.B |
| 数据 | nuScenes 500/100/250，与 MotionNet 同预处理 | p.5 §IV.A |

## 网络（Fig 2, p.3）

| 模块 | 规格 | 出处 |
|---|---|---|
| BEV Encoder | 骨干输出 B∈R^(H×W×D) | p.3 §III.B |
| **IaIE** | N 个实例 query Q∈R^(N×D)（DETR 式可学习）；query 自注意力（式1）→ 与降采样 BEV B'∈R^(Hc×Wc×D) 交叉注意力（式2）→ 实例掩码嵌入 I∈R^(N×D) | p.3 §III.C 式1-2 |
| **IeBE 早期融合** | SE 层精炼 B（式3：F=σ(W·GAP(B))·B）；展平 F̂∈R^(HW×D)；I 过自注意力后用 Hadamard 积+sigmoid 更新：F′=F⊗σ(max_t(F⊗Iᵀ))ᵀ（式4） | p.4 §III.D 式3-4 |
| **IeBE 晚期融合** | 线性投影 φ（式5）后 cross-attn(q=F*, k=v=I*)（式6），FFN 调维 → E | p.4 式5-6 |
| 任务头 | MotionNet 三头 + 掩码头：M̂_i=σ(I_i·B)，类别 MLP 出 Kc+1（含 ∅） | p.4-5 §III.E |
| Transformer 细节 | Pre-LN block、残差、晚期 cross-attention（用户记忆，论文未明写） | notes.md |

## 损失（p.4-5 式7-11）

| 项 | 规格 | 出处 |
|---|---|---|
| Lmot | 加权 SmoothL1，bg w=0.005、其他 1.0 | 式7 |
| Lstate / Lcls | 加权 CE | 式8-9 |
| Lmask-cls | 匈牙利匹配后的 CE（类别）+ 二值掩码损失；L_mask=5.0·BCE+5.0·Dice；匹配代价 C=1.0·(−p_cls)+5.0·BCE+5.0·Dice，no-object 0.1 降权（用户记忆） | 式10 + notes |
| 总 | L=λmot·Lmot+λstate·Lstate+λcls·Lcls+λmask·Lmask-cls（式11） | 式11 |
| **两阶段调度** | 前 20 epoch：λmot=λstate=λcls=1.0、λmask=2.0；后 25 epoch：2.0/2.0/2.0、λmask=1.0 | p.6 §IV.A 原文 |

## 训练配方（p.6）

45 epoch，bs4，单卡 A100，Adam 1.6e-3，衰减 0.5 @ epochs 10/20/30/40——与 R1B 预算一致。

## 评测（p.6）

三档阈值 static ≤0.2 / slow ≤5 / fast >5 m/s；mean+median，**1 秒未来**（与 PriorMotion 口径一致，已由实测验证）；OA 全体非空格均值、MCA 逐类均值（式12-13）。

## 报告数字（Table II/III/IV）

- STPN 骨干：static 0.0228 / slow 0.2294 / **fast 0.9373**（median 0/0.0946/0.6242）vs MotionNet 0.0239/0.2467/1.0109（fast −7.28%）
- 分类：Bg 96.4/Veh 90.6/Ped 81.5/Bike 23.3/Others 75.2，MCA 73.4，OA 95.6
- 参数/延迟：**8.1M / 21ms**（MotionNet 7.9M/19ms，增量 ~0.2M）
- Motion stability：0.0098（vs MotionNet 0.0267）
- 玩具实验（Table I）：GT 类别掩码直接给 MotionNet 作额外输入 → 0.0184/0.2117/0.8369（说明实例信息价值）

## 歧义清单

1. **N 与 D 的数值**：论文未给；用户记忆 N=256、D=64。16×16 降采样 BEV 作 kv 亦为用户记忆（Hc=Wc=16）。
2. **IaIE 的 query 初始化与层数**：DETR 式可学习；层数未给（取 Pre-LN Transformer 单 block 或 L 层待定）。
3. **掩码头与 BEV 特征的交互方式**：式4 的 max_t 操作实现细节需按 notes 记忆（Hadamard+sigmoid）。
4. **Hungarian 匹配的类别代价与掩码项组合**：C=1.0·(−p_cls)+5.0·BCE+5.0·Dice、no-object 0.1（用户记忆）；论文式10 未给系数。
5. **两阶段调度的 λ 作用对象**：原文 "λmot=λstate=λcls=1.0, λmask-cls=2.0"（前 20ep）——直接照抄。
6. GT 实例掩码的来源：nuScenes 3D 框投影到 BEV 的实例图（motionnet_v1 已有 pixel_instance_ids ✓ 数据层零新增）。
