"""扩展评测口径（D8，论文 Table 4/5）。移植自旧代码 eval_general.py：

- 空间分桶：到自车距离 [0,10)/(10,20)/(20,100) 米。
- 误差分解：x/y 分量、径向/切向分量（径向指指向自车，切向垂直）。
- 运动稳定性：逐实例速度（位移模）方差的均值（预测 vs GT 各一份；论文 Motion S. 取预测侧）。
- general 掩码：可选剔除第 5 类（other）格子，用于泛化口径。

指标核心（compute_general_metrics）与模型推理解耦，可用合成数组做 CPU 单测。
协议名：motionnet_v1_ext。
"""
import json
import os

import numpy as np

HEIGHT_FEAT = 13
CELL_CATEGORY_NUM = 5
SPEED_INTERVALS = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0]])
TIER_NAMES = ["static", "slow", "fast"]
CAT_NAMES = {0: "Bg", 1: "Vehicle", 2: "Ped", 3: "Bike", 4: "Others"}
ROI_BORDER = 8
STATIC_UPPER_THD = 0.2
SPATIAL_BINS = [(0, 10), (10, 20), (20, 100)]
GRID_SIZE = 0.25
GRID_CENTER = (128, 128)


def _ego_maps():
    x = np.arange(256) * GRID_SIZE - GRID_CENTER[0] * GRID_SIZE
    y = np.arange(256) * GRID_SIZE - GRID_CENTER[1] * GRID_SIZE
    xx, yy = np.meshgrid(x, y)
    dist = np.sqrt(xx ** 2 + yy ** 2)
    cx = np.full_like(xx, GRID_CENTER[0] * GRID_SIZE)
    cy = np.full_like(yy, GRID_CENTER[1] * GRID_SIZE)
    rx, ry = cx - xx, cy - yy
    rn = np.sqrt(rx ** 2 + ry ** 2) + 1e-8
    return dist, rx / rn, ry / rn


def compute_general_metrics(gt_all, pred_all, valid_all, non_empty_all, inst_all,
                            num_future_sweeps=20, mask_others=False):
    """从逐样本收集的数组算扩展指标。

    gt_all / pred_all: (B, F, 2, H, W) 与 (B, F, 2, H, W)（已含 adj 展平）
    valid_all: (B, F, H, W)；non_empty_all: (B, H, W)；inst_all: (B, H, W) 实例图
    """
    gt_all = np.asarray(gt_all)
    pred_all = np.asarray(pred_all)
    valid_all = np.asarray(valid_all)
    non_empty_all = np.asarray(non_empty_all)
    inst_all = np.asarray(inst_all)

    dist_map, rux, ruy = _ego_maps()
    tux, tuy = -ruy, rux
    selected = np.arange(0, num_future_sweeps + 1, 4)[1:]  # [4,8,12,16,20]
    tier_stats = {t: {s: {k: [] for k in range(len(SPATIAL_BINS))} for s in selected}
                  for t in range(len(SPEED_INTERVALS))}

    gt_norm_all = np.linalg.norm(gt_all, axis=2)  # (B,F,H,W)
    upper_bound = STATIC_UPPER_THD / 20  # 每帧位移上限 0.01m
    static_cell_mask = np.all(gt_norm_all <= upper_bound, axis=1) if gt_norm_all.shape[1] else None

    for b in range(gt_all.shape[0]):
        roi = np.zeros((256, 256), dtype=bool)
        roi[ROI_BORDER:-ROI_BORDER, ROI_BORDER:-ROI_BORDER] = True
        last_gt_norm = gt_norm_all[b, -1]
        for j, d in enumerate(SPEED_INTERVALS):
            if j == 0:
                tier_mask = static_cell_mask[b]
            else:
                tier_mask = (d[0] <= last_gt_norm) & (last_gt_norm < d[1])
            for s_i, s in enumerate(selected):
                cell = (valid_all[b, s - 1] > 0) & roi & tier_mask
                if j > 0:
                    cell &= ~static_cell_mask[b]
                if mask_others:
                    cell &= non_empty_all[b] > 0  # 类别掩码在调用方处理（见 evaluate_general）
                gt = gt_all[b, s - 1]
                pred = pred_all[b, s - 1]
                ex, ey = gt[0] - pred[0], gt[1] - pred[1]
                en = np.sqrt(ex ** 2 + ey ** 2)
                er = ex * rux + ey * ruy
                et = ex * tux + ey * tuy
                for k, (lo, hi) in enumerate(SPATIAL_BINS):
                    m = cell & (dist_map >= lo) & (dist_map < hi)
                    if m.any():
                        tier_stats[j][s][k].append({
                            "norm": en[m], "x": ex[m], "y": ey[m],
                            "radial": er[m], "tangential": et[m],
                        })

    spatial = {}
    for j, name in enumerate(TIER_NAMES):
        spatial[name] = {}
        for s in selected:
            spatial[name][str(s)] = {}
            for k, (lo, hi) in enumerate(SPATIAL_BINS):
                recs = tier_stats[j][s][k]
                if not recs:
                    spatial[name][str(s)][f"{lo}-{hi}"] = None
                    continue
                cat = {key: np.concatenate([r[key] for r in recs]) for key in recs[0]}
                spatial[name][str(s)][f"{lo}-{hi}"] = {
                    "mean": float(np.mean(cat["norm"])),
                    "q50": float(np.quantile(cat["norm"], 0.5)),
                    "x_abs": float(np.mean(np.abs(cat["x"]))),
                    "y_abs": float(np.mean(np.abs(cat["y"]))),
                    "radial_abs": float(np.mean(np.abs(cat["radial"]))),
                    "tangential_abs": float(np.mean(np.abs(cat["tangential"]))),
                }

    pred_var, gt_var = [], []
    for b in range(gt_all.shape[0]):
        inst = inst_all[b]
        gt_last, pred_last = gt_all[b, -1], pred_all[b, -1]
        for iid in np.unique(inst):
            if iid == 0:
                continue
            m = inst == iid
            pred_speed = np.linalg.norm(pred_last[:, m], axis=0)
            gt_speed = np.linalg.norm(gt_last[:, m], axis=0)
            pred_var.append(float(np.var(pred_speed)))
            gt_var.append(float(np.var(gt_speed)))

    stability = {
        "mean_pred_speed_variance": float(np.mean(pred_var)) if pred_var else None,
        "mean_gt_speed_variance": float(np.mean(gt_var)) if gt_var else None,
        "n_instances": len(pred_var),
    }
    return {"spatial": spatial, "stability": stability}


def evaluate_general(model, dataset_root, split="test", out_dir=None, batch_size=1, num_workers=2,
                     device="cuda", use_adj_frame_pred=True, use_motion_state_pred_masking=True,
                     future_frame_skip=0, num_future_sweeps=20, run_id="", ckpt_path="",
                     mask_others=False, meta=None):
    """推理循环复用 motion_eval.evaluate（collect_arrays=True），扩展指标在其上计算。"""
    from bevflow.evaluators.motion_eval import evaluate as evaluate_base

    base, arrays = evaluate_base(
        model, dataset_root, split=split, out_dir=None, batch_size=batch_size,
        num_workers=num_workers, device=device, use_adj_frame_pred=use_adj_frame_pred,
        use_motion_state_pred_masking=use_motion_state_pred_masking,
        future_frame_skip=future_frame_skip, num_future_sweeps=num_future_sweeps,
        run_id=run_id, ckpt_path=ckpt_path, meta=meta, collect_arrays=True,
    )
    ext = compute_general_metrics(
        arrays["gt"], arrays["pred"], arrays["valid"], arrays["non_empty"], arrays["inst"],
        num_future_sweeps=num_future_sweeps, mask_others=mask_others,
    )
    metrics = dict(base)
    metrics["spatial"] = ext["spatial"]
    metrics["stability"] = ext["stability"]
    metrics["meta"] = {**base.get("meta", {}), "eval_protocol": "motionnet_v1_ext"}

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics_general.json"), "w") as f:
            json.dump(metrics, f, indent=1)
    return metrics
