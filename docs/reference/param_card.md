# 参数卡（motionnet_v1 协议）

每个值注明出处。上游 commit `aea30a1` + patches 0001-0003（均为行为等价修补）。

## 数据（出处：third_party/MotionNet/data/gen_data.py）

| 参数 | 值 | 出处 |
|---|---|---|
| 体素/像素大小 | (0.25, 0.25, 0.4) m | gen_data.py:64-66 |
| BEV 范围 | xy ±32 m，z [-3, 2] m | gen_data.py:67 |
| 过去帧数（网络输入） | 5（含当前帧） | gen_data.py:70, convert_to_dense_bev |
| 未来帧数（预测） | 20（1.0 s @ 20 Hz） | out_seq_len=20 |
| past_frame_skip | 3（BEV 序列抽帧间隔） | gen_data.py:68 |
| future_frame_skip | 0 | gen_data.py:69 |
| train 抽帧 | nsweeps_back=30，forward=20，跳关键帧 0，邻接序列 2 | gen_data.py:48-56 |
| val/test 抽帧 | nsweeps_back=25，forward=20，跳关键帧 1，邻接序列 1 | gen_data.py:57-66 |
| 自车补偿 | 多帧点云经 devkit 变换到参考帧坐标系（from_file_multisweep_bf_sample_data） | gen_data.py:97-103 |
| 类别映射 | car/bus→1，ped→2，bike→3，其他→4，背景 0 | gen_data.py:39-45 |
| 相邻序列错开 | 1 个 sample_data（0.05 s），标注取 keyframe | gen_data.py:290-300 |
| split | 上游 data/split.npy，train 500 / val 100 / test 250（已断言） | data/split.npy |

## 模型（出处：third_party/MotionNet/model.py）

| 参数 | 值 | 出处 |
|---|---|---|
| 主干 | STPN：2D 预处理(32ch) → 2×Conv3D 时空块(64/128) → 编码 64/128/256/512 → 解码带 skip + adaptive_max_pool3d | model.py:69-191 |
| 输出头 | 位移 (2×20ch)、格子类别 (5)、动静状态 (2) | model.py:194-220 |
| 参数量 | 论文值 7.9M（官方权重加载后实测，见 reports/R1.md） | 论文 |

## 损失（出处：train_multi_seq.py + README 官方命令）

| 参数 | 值 | 出处 |
|---|---|---|
| 位移损失 | SmoothL1（reduction=none），按类别加权 | train_multi_seq.py:377-383 |
| 背景格子权重 | 0.005，前景 1.0 | train_multi_seq.py:372（weight_vector） |
| 动静分类权重 | 背景 0.005，移动 1.0 | train_multi_seq.py:407 附近 |
| 一致性损失 | bg_tc 0.1，fg_tc 2.5，sc 15.0，全开 | README:31（官方命令）+ argparse 默认 |
| nn_sampling | 关（官方命令未带） | README:31 |

## 优化（出处：train_multi_seq.py）

| 参数 | 值 | 出处 |
|---|---|---|
| 优化器 | Adam，lr 0.0016 | train_multi_seq.py:175 |
| 调度 | MultiStepLR milestones [10,20,30,40]，gamma 0.5 | train_multi_seq.py:176 |
| 调度顺序 | scheduler.step() 在 epoch 开始（上游固有，保持） | train_multi_seq.py:195 |
| epoch | 45 | README:31 + argparse:71 |
| batch size | R1A=8（上游原配方）；R1B=4（项目预算，46GB 卡实测峰值 5.23GiB） | README:31 + configs/exp/ |
| 种子 | 主表 3 个（0,1,2），R1A/R1B 先跑种子 0 | 项目协议 |

## 评测（出处：eval.py；协议名 motionnet_v1）

| 参数 | 值 | 出处 |
|---|---|---|
| 速度三档 | static [0,0]，slow (0,5]，fast (5,20] m/s | eval.py:100 |
| 动静门控 | 0.2 m/s，frame_skip=0 → 每帧位移 ≤0.01 m | eval.py:199-200 |
| 评测帧 | 第 4,8,12,16,20 帧（0.2–1.0 s） | eval.py:101-104 |
| ROI | 裁边 8 像素（2 m） | eval.py:222-224 |
| EPE 聚合 | 5 个评测帧均值/中位数的平均；逐帧数组保留在 metrics.json | 本项目协议（smoke.md） |
| OA | 非空 ∩ one-hot ∩ ROI 格子上预测类别正确比例；上游按 batch 累计，bs=1 才严格逐样本 | eval.py:255-257, 312 |
| MCA | 5 类准确率均值（缺失类 nan，官方权重评测在 test 上不会缺） | eval.py:320-326 |
| 预测后处理 | 动静状态掩码 + 背景类掩码 + 末帧位移 ≤0.2 m 归零（默认开） | eval.py:158-178 |
| ckpt 挑选 | 用 val；规则见 configs，报告同时给 last 与 last-10 均值±标准差 | 项目协议 |

## 与项目笔记固定值的核对

- 背景权重 0.005：与上游一致 ✓
- 动静阈值 0.2 m/s：与上游一致 ✓
- 45 epoch：与上游默认一致 ✓；bs4 是项目预算（R1B），上游原配方 bs8 保留给 R1A ✓
- 主表 3 种子：项目协议，上游无对应概念
