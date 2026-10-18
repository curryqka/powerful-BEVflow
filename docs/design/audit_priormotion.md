# audit_priormotion（C2 旧代码审计）

追溯表：规格条目（spec_priormotion.md）vs 旧代码（ref/priormotion/code/ = powerfulmotion/）。状态：一致 / 不一致 / 缺失 / 多余 / 不确定。数据层专项在末节。

## 追溯表

| 规格条目 | 旧代码位置 | 状态 | 说明 |
|---|---|---|---|
| STPN 骨干 | model/priormotion.py:140-264 | 一致（改动） | 结构与上游同；forward 返回三元组 (res_x, x_6, x_7)，x_6 作 SGRU 隐态——论文未提此细节，属实现需要 |
| 三头 CellCls/StateCls | priormotion.py:15-42 | 一致 | 与 MotionNet 逐字节相同 |
| MotionPredictionV2（单帧 2ch） | priormotion.py:59-83 | 不确定 | 论文式9 的 FSD 带 SE 层（附录）；代码无 SE。自回归逐帧解码与式9 一致 |
| SimpleVAEV2（先验/后验编码） | model/vae_model.py:562-730 | 不一致（结构） | 代码 = present/future 两个条件 VAE（空间隐变量 32×16×16，log_var clamp ±5）；论文 Fig 2 = Dynamic/Static 双 VAE。哪个产出论文数字待裁决（D2） |
| future VAE 输入 = [特征, scene_gt] | priormotion.py:282-283, 306-309 | 不确定 | scene_gt = cat(cls 5ch, disp 2T, motion 2)——论文未描述此拼接；teacher-forcing 机制论文未描述 |
| KL 闭式（future‖present） | train_VAE.py:1001-1019 | 不一致（方向） | 论文式12 为 KL(p(Z\|B)‖p(Z\|B_prior))；代码 KL(future‖present)——present=观测分支≈B，future=GT 条件分支≈B_prior，方向实际一致，表述不同 |
| SGRU + FuturePrediction | model/layers.py:215-267, priormotion.py:84-111 | 一致 | 与式9 逐帧生成吻合；latent repeat 4×4×4 上采样代码特有 |
| 分类解码 ⊕ | priormotion.py:331, 336 | 不确定 | 代码 = latent repeat 上采样 + x 残加；conv_fuser 存在但被注释（:332）。式11 未定义 ⊕ |
| RpE（栅格先验） | — | **缺失** | 无任何对应实现（TSA/SSA/ρ 均无） |
| VpE（矢量先验） | model/prior_extractor.py（GridInsTransformer=ISA:99、位置编码、DispEncoding:89） | 不确定 | 积木在但主模型未接线；完整接线只存在于未启用的 UniGRUMotionNetVAE（en_model.py:718）；LSTM 逐实例编码在 data/instance_extractor.py:277 配套 |
| PCA（先验交叉注意力） | prior_extractor.py:259 MaskedCrossAttention | 不确定 | 同上，积木在未接线 |
| 损失 disp/cls/motion | train_VAE.py:1021-1171 | 一致 | 与 MotionNet 逐行同构（权重 0.005/1.0 同），差异：backward 外置、返回 loss_aux |
| 损失 bg_tc/fg_tc/sc | train_VAE.py 同上 + 脚本 | 一致（权重） | bg_tc 0.1、fg_tc 2.5、**sc 2.0**（上游 MotionNet 为 15.0——旧代码已自行降权） |
| KL 权重与总合成 | train_VAE.py:537-539 | 一致 | 总损失 = 5.0·aux + 0.5·KL；论文式13 的 λ 数值缺失，代码即答案 |
| 训练配方 bs8/60ep（脚本） | run_priormotionstpn.sh | 不一致 | 论文 §4.1 bs4/45ep（nuScenes）；脚本为 wt128 私有数据线。nuScenes 命令未留存 |
| 数据 nuScenes 线 | data/gen_data.py | 一致 | 与上游逐字段等价（"qka 20231204" 仅格式注释）→ motionnet_v1 直接可用 |
| 数据 wt128/清华线 | data/gen_Hdata.py、gen_tsdata.py、HeTools/ | 多余 | 私有传感器线，依赖 haplugin（非 PyPI），本项目不复现 |
| 评测三档+聚合 | eval.py:234 起 | 一致 | 与上游 eval.py 完全同构；我们 motion_eval 可直接用 |
| 评测扩展口径 | eval_general.py | 多余（有用） | 空间分桶/误差分解/实例稳定性/--general 掩码——论文 Table 4/5 的口径，P6 后段按需移植 |
| **多余项** | train.py（坏）、" copy" 文件、train_*_MGDA.py、min_norm_solvers.py、gan_model.py、tsinghua_model.py、be_sti.py、HeTools、plots*/visual* | 多余 | 遗留/备份/私有数据线；不搬 |

## 数据层专项

| 项 | 结论 |
|---|---|
| 输入帧数与时间对齐 | 5 帧（4 过去 + 当前），与 motionnet_v1 相同。旧代码 TrainDatasetMultiSeq 同构 |
| 自车补偿与坐标系 | 多帧点云经 devkit 变换到参考帧（上游机制），未改 |
| 体素化范围/分辨率/高度 bin | 0.25/0.25/0.4，[-32,32]²，z[-3,2]，13 bins——与 motionnet_v1 相同。坑：utlis/params_utlis.py:5 的 height_feat_size=26 为 wt128 残留，不采信 |
| 标签生成 | disp_field/valid_pixel_map/pixel_cat_map/motion 上游同构；pixel_instance_ids 上游本就有 |
| 额外数据产物 | **nuScenes 线无**。raster 先验 = 训练时由 GT 现场构造（scene_gt），不落盘；wt128 线的速度/方向栅格为本项目不复现的私有数据 |
| 划分 | 同一 data/split.npy 500/100/250 |
| 增广 | 无（论文 §4.2 明示） |
| 评测代码差异 | 三档阈值/聚合/ROI 与上游一致；eval_general 增空间分桶、误差分解、实例稳定性、--general 掩码（论文新指标），P6 后段移植 |
| **数据层总判定** | **motionnet_v1 直接可用，零新数据工作**。旧代码 nuScenes 数据线无错误遗传 |

## 审计结论

旧代码 = DSpG 的可信参考实现（Table 7(e) 档）+ RVpE 的半成品积木。论文主结果（Table 2 的 0.8324）所需的完整模型在仓库中不存在，需按论文实现 RVpE 并接线。旧代码"能跑出好结果的 trick"清单：sc 降权 2.0（论文未写）、总损失 5.0 加权（论文未写）、log_var clamp ±5（论文未写）、teacher-forcing 替换（论文未描述）——均已列进决策清单。
