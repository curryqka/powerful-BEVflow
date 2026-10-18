# spec_priormotion（C1 规格提取）

来源：ref/priormotion/paper.pdf（ICCV 2025 open access，11 页）。每条标页码/公式号/表号。歧义单列末节。

## 任务设定

| 项 | 规格 | 出处 |
|---|---|---|
| 输入 | 5 帧体素化点云 V∈{0,1}^(H×W×C)，0.25m×0.25m×0.4m，范围 [-32,32]²×[-3,2]，最后一帧为当前时刻 | p.6 §4.1 |
| 输出 | 运动场 M∈R^(H×W×2)×T（T=20），格子分类 C∈R^(H×W×5)，状态 S∈R^(H×W) | p.3 §3.1 |
| 数据划分 | nuScenes 500/100/250 | p.6 §4.1 |
| 骨干 | STPN（MotionNet）或 BE-STI，可插拔；本项目复现 STPN 版 | p.3 §3.2.1, Table 2 |

## 网络（Fig 2, p.3；Fig 3, p.4；Fig 4, p.5）

| 模块 | 规格 | 出处 |
|---|---|---|
| BEV Encoder | STPN 输出 B∈R^(H×W×C′) | p.3 §3.2.1 |
| RpE（栅格先验，仅训练） | 局部支：1×1 3DConv(M) + 1×1 2DConv([C,S])（式2）；全局支：M 降分辨率→TSA→与 [C,S] 过 SSA（式3）；可学习因子 ρ 融合 P_R=ρ·F_global+(1−ρ)·F_local（式4） | p.4 §3.2.2 |
| VpE（矢量先验，仅训练） | N_ins 个实例序列，每实例 N 个轨迹网格点：(x,y) 正弦位置编码 + 位移向量 + one-hot 类别；D_I=N·d_pos+N_C+T·d_M（式5）；LSTM 编码时序；L=6 层 Transformer 自注意力 ISA → P_V（式6） | p.4 §3.2.2 |
| Prior fusion | B_prior=[B, P_R, PCA(q=B, kv=P_V)]（式7），PCA=Prior Cross-Attention | p.5 §3.2.2 |
| DSpG 先验建模 | 动/静掩码来自现成感知模型；双 VAE 编码 B_dyn/B_static → p(Z\|M)~N(μ_f,σ_f²)；Z∈R^(H/4×W/4×D_Z) 空间下采样 | p.5 §3.2.3 式8 |
| 潜空间生成 | SGRU 逐帧：Z_{t+1}=SGRU(Z_t)，M_{t+1}=F_SD(Z_{t+1})（FSD 带 SE 层，附录）；联合条件概率式10 | p.5 式9-10 |
| 分类解码 | p(C,S\|Z_0)=B′⊕Z_0（特征融合，⊕具体方式论文未给） | p.5 式11 |
| 三头 | Cell Classification / Motion Prediction / State Estimation，沿 MotionNet 惯例 | p.5（"standard practices from [51]"） |

## 损失（p.5 §3.2.4）

| 项 | 规格 | 出处 |
|---|---|---|
| L_prior | KL(p(Z\|B) ‖ p(Z\|B_prior))（式12） | p.5 式12 |
| 总损失 | L=λ_mot·L_mot+λ_state·L_state+λ_cls·L_cls+λ_prior·L_prior（式13） | p.5 式13 |
| λ 数值 | 论文正文未给，"details in the appendix"——**本项目无附录 PDF，以旧代码为准** | p.5 §3.2.4 |

## 训练配方（p.6 §4.1）

Adam lr 0.0016；衰减 0.5 @ epochs 10/20/30/40；45 epoch；**bs4 单卡 A100**；同 MotionNet 预处理；无数据增强；无 MGDA（p.6 §4.2 "For fairness, no data augmentation or multi-gradient descent algorithm (MGDA) is used"）。

## 评测（p.6 §4.1）

MotionNet 协议：非空格子三档（static ≤0.2 m/s，slow ≤5，fast >5），mean+median；OA/MCA。**已实证口径为 1.0s 单点值**（官方权重 slow@1.0s=0.247 vs 论文 0.2467，见 baseline_numbers.md）。新指标：泛化 GI（式14，掩"other"类训练）、运动稳定性（式15，实例内位移方差）、长距离分桶 [0,10]/[10,20]/[20,∞)。

## 报告数字（全部表）

- Table 2（p.6）：STPN/w(Ours) **0.0251/0.2247/0.8324**（对照 MotionNet 原版 0.0262/0.2467/0.9878、MotionNet†MGDA 0.0201/0.2292/0.9454）
- Table 3（p.7）：分类 Bg 94.6/Veh 92.1/Ped 86.9/Bike 27.3/Others 80.4，MCA 76.3，OA 93.6（OA 低于 MotionNet 96.3）
- Table 4（p.7）：GI 86.4%（STPN/w Ours）
- Table 5（p.7）：长距离 static 0.0263/slow 0.2207/fast 0.8549，Motion S. 0.0088
- Table 6（p.7）：(d) 11.5M，69ms（12ms 体素化+57ms 前向，RTX 3090），0.0251/0.2247/0.8318
- Table 7（p.8）：消融 (a) VpE 0.9580 / (b) RpE 0.9351 / (c) VpE+RpE 0.9028 / (d) DSpM 0.9564 / (e) DSpM+LFmG 0.8943 / (f) 全配置 0.8324（fast 档 Mean）

## 消融验证目标

Table 7 的方向：VpE+RpE 逐级降 fast 误差；DSpM 防静态退化；DSpM+LFmG 叠加最大收益（0.9564→0.8324）。

## 效率

11.5M 参数；69ms（12ms 体素化 + 57ms 前向，RTX 3090）——Table 6(d)。

## 歧义清单

1. **λ 四项权重数值**：正文无，附录缺失 → 以旧代码为准（aux 5.0 总加权 / kld 0.5 / bg_tc 0.1 / fg_tc 2.5 / sc 2.0）。
2. **双 VAE vs 条件 VAE**：论文 Fig 2 画 Dynamic VAE + Static VAE（动/静分解）；代码是 present/future 条件 VAE（动/静掩码未出现在代码路径）。哪个产出论文数字待裁决（D2）。
3. **RVpE 的实现**：论文 Fig 3 完整（RpE 全局/局部+ρ、VpE LSTM+ISA）；代码只有积木（prior_extractor.py 的 GridInsTransformer/MaskedCrossAttention/位置编码），RpE 无实现、主模型未接线（D3）。
4. **FSD 的 SE 层**：论文说带 SE layer（附录），正文无细节；代码 MotionPredictionV2 无 SE → 按代码（D4 候选）。
5. **动/静掩码来源**：论文说 "off-the-shelf perception model"；代码用 GT 的 cls/motion 图作 scene_gt → 待裁决（D5）。
6. **B′⊕Z_0 融合方式**：式11 未给 ⊕ 定义；代码 = latent 重复上采样 + x 残加（conv_fuser 存在但被注释）（D6 候选）。
7. **fast@1.0s = 0.8324（Table 2）vs 0.8318（Table 6d）**：两表微差，取 Table 2。
