# PriorMotion 盘点（P6，2026-10-18）

来源：旧代码 `third_party/MotionNet/powerfulmotion/`（87 py，MotionNet fork 改造，无 git 历史，mtime 全为拷贝时刻）+ ICCV 2025 论文。探查方式：全量交叉验证（import 关系、pycache、shell 调用链）。

## 真实训练入口与生效模型

- **训练入口 = `train_VAE.py`**（全仓唯一构造 PriorMotionSTPN 的顶层脚本；`train.py` 是拼接遗留已坏；" copy" 文件逐字节等于正本）。启动脚本 `scripts/run_priormotionstpn.sh`。
- **生效模型 = `model/priormotion.py` 的 PriorMotionSTPN**（PriorMotionSTI 为换骨干变体），解析参数量 ≈11.5–12.5M，与论文 Table 6(d) 的 11.5M 吻合 → 论文数字出自此版本代码。
- 超参（脚本实际传入）：bs8，lr 0.0016，nepoch 60（脚本覆盖 argparse 默认 45），Adam + MultiStepLR[10,20,30,40]×0.5，损失权重 re_weight=5.0、kld=0.5、bg_tc=0.1、fg_tc=2.5、**sc=2.0**（注意：不是上游 MotionNet 的 15.0）。
- 数据路径写死 `all_train240_val30_wt128_0`（禾赛 wt128_0 私有数据，240/30 场景）。**nuScenes 版训练命令未留存**（logs/ 里有 MotionNet_nuscenes、GRUMotionNetVAE_nuscenes、unigen_nuscenes 的评测 txt，说明跑过）。

## 模型结构（代码为准，与论文有出入）

```
PriorMotionSTPN
├── stpn: STPN（改 forward 返回 (res_x, x_6, x_7)，x_6 做 SGRU 隐态初始化）
├── 三头: cell_classify / motion_pred=MotionPredictionV2(单帧 2ch 输出！非 MotionNet 的 2×20ch)
│         / state_classify
├── present_distrubution: SimpleVAEV2(32→latent 32, 16×16×32 空间隐变量)   ← 观测先验（推理用）
├── future_distrubution: SimpleVAEV2(32+47→32)                            ← 未来后验（训练用）
│     输入 = [骨干特征, scene_gt = cat(cls 5ch, disp 2T ch, motion 2ch)]
└── future_prediction: FuturePrediction(3×SpatialGRU + Bottleneck)        ← 自回归解码 T 帧
```

- 条件 VAE 机制：训练时未来后验采样**直接替换**观测先验的 bev_refine（teacher-forcing）；推理时 disp_gt_map=None，从 present 分支 reparameterize 采样。KL = 闭式 KL(future‖present)，逐位置求和，权重 0.5，无 annealing。
- SimpleVAEV2 无解码器——"解码" = FuturePrediction + MotionPredictionV2 逐帧。
- **与论文的差异**：论文 Fig 2 画的是 Dynamic VAE + Static VAE 双分支 + RVpE（RpE raster 全局/局部双支 + VpE LSTM/ISA 矢量先验）；代码 PriorMotionSTPN **没有** RpE/VpE 模块（VpE 的 ISA 只存在于未启用的 UniGRUMotionNetVAE 演化支线）。Table 7(f) 全配置（VpE+RpE+DSpM+LFmG，fast=0.8324）与 Table 6(d) 11.5M/0.8318 的对应代码在仓库中**不完整**——DSpG 部分齐全，RVpE 部分缺失。

## 数据

- **nuScenes 线 gen_data 与上游逐字段等价**（pixel_instance_ids 上游本就有）→ **motionnet_v1 直接可用，无需新数据生产**。
- wt128/清华线（gen_Hdata、voxelize_with_velocity 速度/方向先验栅格、haplugin 依赖）不搬——用户确认数据按 MotionNet。

## 评测

- 三档阈值、聚合（pool-then-average + q50）、ROI 与上游完全一致 → **bevflow/evaluators/motion_eval.py 直接可用**。
- 论文 Table 4/5 的扩展口径在 `eval_general.py`：空间分桶 (0,10)/(10,20)/(20,100)m、误差 x/y/radial/tangential 分解、逐实例速度稳定性、--general 掩第 5 类泛化。logs/ 里 39 个 eval txt 全是它的输出。
- **旧 ckpt 全盘为零**——复现只能重训。

## 移植映射（PriorMotion 增量）

| 旧组件 | 新位置 |
|---|---|
| priormotion.py: STPN(3 返回值)/三头/MotionPredictionV2/FuturePrediction/PriorMotionSTPN | `bevflow/models/priormotion.py` + registry 注册 |
| vae_model.py: SimpleVAEV2（仅此一个） | `bevflow/models/vae.py` |
| layers.py: SpatialGRU/Bottleneck | `bevflow/models/layers.py` |
| train_VAE.py compute_aux_loss / compute_vae_loss / train() 的 scene_gt teacher-forcing | `models/losses.py` 增 kld 项 + `trainers/trainer_prior.py`（trainer 层唯一结构性改动：GT 传 forward） |
| nuscenes_dataloader 两 Dataset | 已移植类，无需扩展（nuScenes 线无新字段） |
| eval_general.py 扩展口径 | `evaluators/motion_eval_general.py`（P6 后段，论文 Table 4/5 对照用） |
| en_model/tsinghua_model/gan/be_sti/HeTools/wt128 全线 | 不搬 |

## 三个已识别的坑

1. utlis/params_utlis.py 的 height_feat_size=26 与 eval/dataloader 的 13 矛盾——nuScenes 线以 13 为准（预处理实测 256×256×13）。
2. 速度先验通道拼在 BEV z 维（height+1/+2）——nuScenes 线不用（无多普勒）。
3. teacher-forcing（训练）与纯先验采样（推理）的不对称是 KL 成立的前提，trainer 移植不可丢。
