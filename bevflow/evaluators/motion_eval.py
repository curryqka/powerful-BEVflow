"""位移评测与分类评测。从上游 eval.py 的 eval_motion_displacement 搬运。

- 协议 motionnet_v1：三档 static/slow/fast（0.2 m/s 门控，5/20 m/s 分箱），
  评测帧 [4,8,12,16,20]（0.2s 间隔），ROI 裁边 8 像素。
- metrics.json 直接在此写出（吸收原 scripts/eval_parse.py），txt 保持上游格式可对照。
- 上游怪癖保持：OA 按 batch 累计再除以样本数，严格逐样本均值需 bs=1（README 官方用法）。
  混淆矩阵显式给 5 类标签（对应上游补丁 0003）。
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from bevflow.datasets.motionnet import DatasetSingleSeq

HEIGHT_FEAT = 13
CELL_CATEGORY_NUM = 5
SPEED_INTERVALS = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0]])  # m/s
TIER_NAMES = ["static", "slow", "fast"]
CAT_NAMES = {0: "Bg", 1: "Vehicle", 2: "Ped", 3: "Bike", 4: "Others"}
ROI_BORDER = 8
STATIC_UPPER_THD = 0.2  # m/s


def evaluate(
    model,
    dataset_root,
    split="val",
    out_dir=None,
    batch_size=1,
    num_workers=2,
    device="cuda",
    use_adj_frame_pred=True,
    use_motion_state_pred_masking=True,
    future_frame_skip=0,
    num_future_sweeps=20,
    run_id="",
    ckpt_path="",
    meta=None,
):
    dataset = DatasetSingleSeq(
        dataset_root=dataset_root,
        split=split,
        future_frame_skip=future_frame_skip,
        voxel_size=(0.25, 0.25, 0.4),
        area_extents=np.array([[-32.0, 32.0], [-32.0, 32.0], [-3.0, 2.0]]),
        num_category=CELL_CATEGORY_NUM,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    speed_intervals = SPEED_INTERVALS
    selected_future_sweeps = np.arange(0, num_future_sweeps + 1, 3 + 1)[1:]  # [4, 8, 12, 16, 20]
    last_future_sweep_id = selected_future_sweeps[-1]
    distance_intervals = speed_intervals * (last_future_sweep_id / 20.0)

    cell_groups = [[[] for _ in selected_future_sweeps] for _ in distance_intervals]

    model.eval()
    pixel_acc = 0.0
    overall_cls_gt = list()
    overall_cls_pred = list()

    for data in dataloader:
        (
            padded_voxel_points,
            all_disp_field_gt,
            all_valid_pixel_maps,
            non_empty_map,
            pixel_cat_map_gt,
            past_steps,
            future_steps,
            motion_gt,
        ) = data

        padded_voxel_points = padded_voxel_points.to(device)
        with torch.no_grad():
            disp_pred, class_pred, motion_pred = model(padded_voxel_points)

            pred_shape = disp_pred.size()
            disp_pred = disp_pred.view(all_disp_field_gt.size(0), -1, pred_shape[-3], pred_shape[-2], pred_shape[-1])
            disp_pred = disp_pred.contiguous().cpu().numpy()

            if use_adj_frame_pred:
                for c in range(1, disp_pred.shape[1]):
                    disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]

            if use_motion_state_pred_masking:
                motion_pred_numpy = np.argmax(motion_pred.cpu().numpy(), axis=1)
                mask = motion_pred_numpy == 0

                class_pred_cat = np.argmax(class_pred.cpu().numpy(), axis=1)
                class_mask = class_pred_cat == 0

                last_pred = disp_pred[:, -1, :, :, :]
                # disp_pred 布局 (B, F, 2, h, w)：沿通道维取模，得 (B, h, w)（与上游一致）
                last_pred_norm = np.linalg.norm(last_pred, ord=2, axis=1)
                thd_mask = last_pred_norm <= 0.2

                cat_weight_map = np.ones_like(class_pred_cat, dtype=np.float32)
                cat_weight_map[mask] = 0.0
                cat_weight_map[class_mask] = 0.0
                cat_weight_map[thd_mask] = 0.0
                cat_weight_map = cat_weight_map[:, np.newaxis, np.newaxis, ...]

                disp_pred = disp_pred * cat_weight_map

        all_disp_field_gt = all_disp_field_gt.numpy()
        future_steps_np = future_steps.numpy()[0]

        valid_pixel_maps = all_valid_pixel_maps[:, -future_steps_np:, ...].contiguous().numpy()
        all_disp_field_gt = all_disp_field_gt[:, -future_steps_np:]
        all_disp_field_gt = np.transpose(all_disp_field_gt, (0, 1, 4, 2, 3))
        all_disp_field_gt_norm = np.linalg.norm(all_disp_field_gt, ord=2, axis=2)

        upper_bound = (future_frame_skip + 1) / 20 * STATIC_UPPER_THD
        static_cell_mask = all_disp_field_gt_norm <= upper_bound
        static_cell_mask = np.all(static_cell_mask, axis=1)
        moving_cell_mask = np.logical_not(static_cell_mask)

        for j, d in enumerate(distance_intervals):
            for slot, s in enumerate(selected_future_sweeps - 1):
                curr_valid_pixel_map = valid_pixel_maps[:, s]

                if j == 0:
                    curr_mask = np.logical_and(curr_valid_pixel_map, static_cell_mask)
                else:
                    last_gt_norm = all_disp_field_gt_norm[:, -1]
                    mask = np.logical_and(d[0] <= last_gt_norm, last_gt_norm < d[1])
                    curr_mask = np.logical_and(curr_valid_pixel_map, mask)
                    curr_mask = np.logical_and(curr_mask, moving_cell_mask)

                roi_mask = np.zeros_like(curr_mask, dtype=bool)
                roi_mask[:, ROI_BORDER:-ROI_BORDER, ROI_BORDER:-ROI_BORDER] = True
                curr_mask = np.logical_and(curr_mask, roi_mask)

                cell_idx = np.where(curr_mask == True)  # noqa: E712  保持上游写法

                gt = all_disp_field_gt[:, s]
                pred = disp_pred[:, s]
                norm_error = np.linalg.norm(gt - pred, ord=2, axis=1)

                cell_groups[j][slot].append(norm_error[cell_idx])

        # -- 分类准确率
        pixel_cat_map_gt_numpy = pixel_cat_map_gt.numpy()
        non_empty_map_numpy = non_empty_map.numpy()
        class_pred_numpy = class_pred.cpu().numpy()

        max_prob = np.amax(pixel_cat_map_gt_numpy, axis=-1)
        filter_mask = max_prob == 1.0
        pixel_cat_map_numpy = (np.argmax(pixel_cat_map_gt_numpy, axis=-1) + 1) * non_empty_map_numpy * filter_mask
        pixel_cat_map_numpy = pixel_cat_map_numpy.astype(int)

        class_pred_numpy = np.transpose(class_pred_numpy, (0, 2, 3, 1))
        class_pred_numpy = (np.argmax(class_pred_numpy, axis=-1) + 1) * non_empty_map_numpy * filter_mask
        class_pred_numpy = class_pred_numpy.astype(int)

        roi_mask = np.zeros_like(non_empty_map_numpy)
        roi_mask[:, ROI_BORDER:-ROI_BORDER, ROI_BORDER:-ROI_BORDER] = 1.0

        tmp = pixel_cat_map_numpy == class_pred_numpy
        denominator = np.sum(non_empty_map_numpy * filter_mask * roi_mask)
        pixel_acc += np.sum(tmp * non_empty_map_numpy * filter_mask * roi_mask) / denominator

        count_mask = non_empty_map_numpy * filter_mask * roi_mask
        idx_fg = np.where(count_mask > 0)
        overall_cls_gt.append(pixel_cat_map_numpy[idx_fg])
        overall_cls_pred.append(class_pred_numpy[idx_fg])

    # -- 汇总
    txt_lines = list()
    dump_res = list()

    epe = {}
    for i, d in enumerate(distance_intervals):
        section = ["-" * 62, f"For cells within speed range [{d[0]}, {d[1]}]:\n"]
        per_frame_mean, per_frame_median = [], []
        for s in range(len(selected_future_sweeps)):
            errors = np.concatenate(cell_groups[i][s]) if len(cell_groups[i][s]) != 0 else cell_groups[i][s]
            if len(errors) == 0:
                mean_error, quantile_50 = None, None
            else:
                mean_error = float(np.average(errors))
                quantile_50 = float(np.quantile(errors, 0.5))
            per_frame_mean.append(mean_error)
            per_frame_median.append(quantile_50)
            section.append(f"Frame {selected_future_sweeps[s]}:\nThe mean error is {mean_error}\n"
                           f"The 50% error quantile is {quantile_50}")
        section.append("-" * 62)
        txt_lines.extend(section)
        txt_lines.append("")
        epe[TIER_NAMES[i]] = {
            "mean": float(np.mean([v for v in per_frame_mean if v is not None])) if any(
                v is not None for v in per_frame_mean) else None,
            "median": float(np.mean([v for v in per_frame_median if v is not None])) if any(
                v is not None for v in per_frame_median) else None,
            "per_frame_mean": per_frame_mean,
            "per_frame_median": per_frame_median,
            "range_ms": [float(d[0]), float(d[1])],
        }
        dump_res.extend([v for pair in zip(per_frame_mean, per_frame_median) for v in pair])

    oa = pixel_acc / len(dataset)
    overall_cls_gt = np.concatenate(overall_cls_gt)
    overall_cls_pred = np.concatenate(overall_cls_pred)
    cm = confusion_matrix(overall_cls_gt, overall_cls_pred, labels=list(range(5)))
    cm_sum = np.sum(cm, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_cat = cm[np.arange(5), np.arange(5)] / cm_sum
    per_class = {CAT_NAMES[i]: (None if np.isnan(mean_cat[i]) else float(mean_cat[i])) for i in range(5)}
    mca = None if np.isnan(np.mean(mean_cat)) else float(np.mean(mean_cat))

    txt_lines.append(f"Mean pixel classification accuracy: {oa}")
    for i in range(5):
        txt_lines.append(f"mean cat accuracy of {CAT_NAMES[i]}: {mean_cat[i]}")
    txt_lines.append(f"mean instance acc: {np.mean(mean_cat)}")
    txt_lines.append("")
    txt_lines.append("Reshape the displacement prediction statistics into a column:")
    txt_lines.extend(str(item) for item in dump_res)

    metrics = {
        "run_id": run_id,
        "ckpt": ckpt_path,
        "split": split,
        "n_samples": len(dataset),
        "epe": epe,
        "cls": {"OA": float(oa), "MCA": mca, "per_class": per_class},
        "meta": {"eval_protocol": "motionnet_v1", **(meta or {})},
    }

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "eval_displacement.txt"), "w") as f:
            f.write("\n".join(txt_lines) + "\n")
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=1)

    return metrics
