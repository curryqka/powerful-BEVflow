# 模块接口

给后续新方法（P6 复现与第一批实验）用的最小接口清单。

## models

- `MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=13)`，forward(bevs) → (disp, cell_cls, motion_cls)。
- `bevflow/models/loader.load_motionnet(ckpt_path, device)`：加载官方权重或训练 ckpt。
- registry：`bevflow/utils/registry.py` 的 Registry 类，按名构建。

## losses

- `LossConfig`：use_bg_tc / use_fg_tc / use_sc / nn_sampling / reg_weight_*。
- `compute_losses(cfg, disp_gt, valid, cat, disp_pred, non_empty, class_pred, motion_gt, motion_pred, trans, inst, device)`
  → (total 张量或 None, [disp, cls, motion, bg_tc, sc, fg_tc] float 列表)。
  纯函数，backward 与 step 在 trainer。total 为 None 表示该 batch 无有效像素。

## datasets

- `TrainDatasetMultiSeq(dataset_root, ...)`：子目录结构，一致性损失用。
- `DatasetSingleSeq(dataset_root, split, ...)`：扁平 npy 目录，评测用。
- 两者都输出 numpy 数组元组，DataLoader 默认 collate。

## preprocess

- `generate.generate_scenes(nusc, scene_ids, split_kind, savepath)`：场景子集可控。
- `convert.convert_to_dense_bev / convert_to_sparse_bev`：组装与稀疏化。
- `grid.gen_2d_grid_gt / voxelize_occupy / classify_speed_level`。

## evaluators

- `motion_eval.evaluate(model, dataset_root, split, out_dir, batch_size, ...)` → metrics dict，同时写 metrics.json 与上游格式 txt。
- `viz.visualize_samples(model, dataset, idxs, out_dir, seed, ckpt)` → 写 PNG 与 viz_manifest.json。

## trainers

- `multi_seq.train(cfg, run_dir)`：cfg 来自 configs（见 base.yaml 注释）。ckpt 策略：epoch 1、每 5、>20 全存，另有 last.pt。resume auto 从最新 epoch ckpt 续。

## P6 预留（现在只有接口位，不实现）

- 新模型：在 bevflow/models 加模块 + registry 注册，配置里换 name。
- 新指标：evaluators 加函数，metrics.json 的 meta.eval_protocol 标新版本。
