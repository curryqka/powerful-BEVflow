# 上游文件到新位置的映射

上游 = `third_party/MotionNet`（固定 aea30a1 + patches 0001-0003）。搬运原则：数值行为不变，golden 对拍见 tests/。

| 上游内容 | 新位置 | 说明 |
|---|---|---|
| `model.py`（MotionNet/STPN/三头） | `bevflow/models/motionnet.py` | 类名与属性名不变，state_dict 键名兼容官方权重。FeatEncoder/MotionNetMGDA 不搬（MGDA 排除） |
| `train_multi_seq.py` 的 `compute_and_bp_loss` 与两个一致性损失 | `bevflow/models/losses.py` | 模块级全局开关改为 LossConfig；zero_grad/backward/step 移到 trainer；数值路径不变 |
| `data/nuscenes_dataloader.py` 两个 Dataset | `bevflow/datasets/motionnet.py` | 逻辑逐行保持，含 Manager 缓存 |
| `data/data_utils.py` 的 `gen_2d_grid_gt`、`voxelize_occupy`、`point_in_hull_fast`、`calc_displace_vector`、`classify_speed_level` | `bevflow/preprocess/grid.py` | 逐行搬运 |
| `data/gen_data.py` 主循环 + `convert_to_dense_bev` + `convert_to_sparse_bev` | `bevflow/preprocess/generate.py` + `convert.py` | 场景子集原生支持；实例顺序确定化（见下） |
| `train_multi_seq.py` 训练循环 | `bevflow/trainers/multi_seq.py` | 加 train.jsonl / run_manifest / resume auto / seed / tb |
| `eval.py` 的 `eval_motion_displacement` | `bevflow/evaluators/motion_eval.py` | metrics.json 直接产出；混淆矩阵固定 5 类；OA 按 batch 累计的上游怪癖保持（bs=1 才严格逐样本） |
| `plots.py` 的 quiver 可视化 | `bevflow/evaluators/viz.py` | 风格对齐（类别色 c/m/k/y/r、0.4m 显示阈值、4m 裁边）；另加误差图与静态噪点图 |
| `scripts/eval_parse.py`（原 smoke 解析） | 吸收进 `motion_eval.py` | metrics.json 在评测器内直接写出 |
| `data/data_utils.py` 其余（voxelize、pillar 系列、compute_ratio_*、compute_speed_level*、get_static_and_moving_cells） | 不搬 | 体素/支柱变体与调参工具，MotionNet 主路径不经过 |
| `train_single_seq*.py`、`min_norm_solvers.py`、`train_*_MGDA.py` | 不搬 | MGDA 排除；单序列训练不用 |
| `plots.py` 其余（GIF/视频拼装、LIDAR 底图） | 不搬 | 需要时再从原始数据渲染 |

## 与上游的既知差异（均已记录）

1. **实例顺序确定化**（preprocess/generate.py）：上游用 set 迭代重排实例，标签顺序随进程哈希种子变。已实验证实上游重跑自身：位移场 0.01% 像素厘米级差异、实例数 4/34 序列差 1、前景结构始终一致。本管线按 token 排序，两进程逐位一致。下游一致性损失不敏感（标签只用于组内配对）。
2. **补丁 0001-0003**：见 docs/setup/upstream_patches.md。
3. **新增机制**（不影响数值）：seed 固定、train.jsonl、resume auto、每 5 epoch 的 val、tensorboard、metrics.json、run_manifest。
