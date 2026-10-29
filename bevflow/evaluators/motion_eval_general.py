# SPDX-License-Identifier: AGPL-3.0-or-later
# 本文件派生自 MERL MotionNet（https://github.com/merlresearch/MotionNet）。
"""扩展评测口径（D8，论文 Table 4/5）。移植自旧代码 eval_general.py：

- 空间分桶：到自车距离 [0,10)/(10,20)/(20,100) 米。
- 误差分解：x/y 分量、径向/切向分量（径向指指向自车，切向垂直）。
- 运动稳定性：逐实例速度（位移模）方差的均值（预测 vs GT 各一份；论文 Motion S. 取预测侧）。
- general 掩码：可选剔除第 5 类（other）格子。

实现为流式累加器：逐 batch 吸收误差数组（每个元素级误差，非整张 GT/预测体积），
内存与样本数解耦——全量 test（4309 样本）内存 ~1-2GB 而非 ~100GB。

协议名：motionnet_v1_ext。
"""
import json
import os

import numpy as np

SPEED_INTERVALS = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0]])
TIER_NAMES = ["static", "slow", "fast"]
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


class GeneralAccumulator:
    """逐 batch 吸收误差；finalize() 出 spatial/stability 字典。"""

    def __init__(self, num_future_sweeps=20):
        self.selected = np.arange(0, num_future_sweeps + 1, 4)[1:]
        self.dist_map, self.rux, self.ruy = _ego_maps()
        self.tux, self.tuy = -self.ruy, self.rux
        # tier -> frame -> bin -> [error-dict, ...]
        self.tier_stats = {t: {int(s): {k: [] for k in range(len(SPATIAL_BINS))} for s in self.selected}
                           for t in range(len(SPEED_INTERVALS))}
        self.pred_var, self.gt_var = [], []
        self.num_future_sweeps = num_future_sweeps

    def __call__(self, gt_all, pred_all, valid_all, non_empty_all, inst_all):
        gt_all = np.asarray(gt_all)
        pred_all = np.asarray(pred_all)
        gt_norm_all = np.linalg.norm(gt_all, axis=2)
        upper_bound = STATIC_UPPER_THD / 20
        static_mask = np.all(gt_norm_all <= upper_bound, axis=1)
        roi = np.zeros((256, 256), dtype=bool)
        roi[ROI_BORDER:-ROI_BORDER, ROI_BORDER:-ROI_BORDER] = True

        for b in range(gt_all.shape[0]):
            last_gt_norm = gt_norm_all[b, -1]
            for j, d in enumerate(SPEED_INTERVALS):
                tier_mask = static_mask[b] if j == 0 else (d[0] <= last_gt_norm) & (last_gt_norm < d[1])
                for s in self.selected:
                    cell = (valid_all[b, s - 1] > 0) & roi & tier_mask
                    if j > 0:
                        cell &= ~static_mask[b]
                    gt = gt_all[b, s - 1]
                    pred = pred_all[b, s - 1]
                    ex, ey = gt[0] - pred[0], gt[1] - pred[1]
                    en = np.sqrt(ex ** 2 + ey ** 2)
                    er = ex * self.rux + ey * self.ruy
                    et = ex * self.tux + ey * self.tuy
                    for k, (lo, hi) in enumerate(SPATIAL_BINS):
                        m = cell & (self.dist_map >= lo) & (self.dist_map < hi)
                        if m.any():
                            self.tier_stats[j][int(s)][k].append(
                                {"norm": en[m], "x": ex[m], "y": ey[m],
                                 "radial": er[m], "tangential": et[m]})
            inst = inst_all[b]
            gt_last, pred_last = gt_all[b, -1], pred_all[b, -1]
            for iid in np.unique(inst):
                if iid == 0:
                    continue
                m = inst == iid
                if m.sum() == 0:
                    continue
                self.pred_var.append(float(np.var(np.linalg.norm(pred_last[:, m], axis=0))))
                self.gt_var.append(float(np.var(np.linalg.norm(gt_last[:, m], axis=0))))

    def finalize(self):
        spatial = {}
        for j, name in enumerate(TIER_NAMES):
            spatial[name] = {}
            for s in self.selected:
                spatial[name][str(int(s))] = {}
                for k, (lo, hi) in enumerate(SPATIAL_BINS):
                    recs = self.tier_stats[j][int(s)][k]
                    if not recs:
                        spatial[name][str(int(s))][f"{lo}-{hi}"] = None
                        continue
                    cat = {key: np.concatenate([r[key] for r in recs]) for key in recs[0]}
                    spatial[name][str(int(s))][f"{lo}-{hi}"] = {
                        "mean": float(np.mean(cat["norm"])),
                        "q50": float(np.quantile(cat["norm"], 0.5)),
                        "x_abs": float(np.mean(np.abs(cat["x"]))),
                        "y_abs": float(np.mean(np.abs(cat["y"]))),
                        "radial_abs": float(np.mean(np.abs(cat["radial"]))),
                        "tangential_abs": float(np.mean(np.abs(cat["tangential"]))),
                    }
        stability = {
            "mean_pred_speed_variance": float(np.mean(self.pred_var)) if self.pred_var else None,
            "mean_gt_speed_variance": float(np.mean(self.gt_var)) if self.gt_var else None,
            "n_instances": len(self.pred_var),
        }
        return {"spatial": spatial, "stability": stability}

    def add(self, gt_all, pred_all, valid_all, non_empty_all, inst_all):
        self(gt_all, pred_all, valid_all, non_empty_all, inst_all)


def compute_general_metrics(gt_all, pred_all, valid_all, non_empty_all, inst_all,
                            num_future_sweeps=20, mask_others=False):
    """整批数组入口（CPU 单测用）；与流式累加器同一实现。"""
    acc = GeneralAccumulator(num_future_sweeps)
    acc(gt_all, pred_all, valid_all, non_empty_all, inst_all)
    return acc.finalize()


def evaluate_general(model, dataset_root, split="test", out_dir=None, batch_size=1, num_workers=2,
                     device="cuda", use_adj_frame_pred=True, use_motion_state_pred_masking=True,
                     future_frame_skip=0, num_future_sweeps=20, run_id="", ckpt_path="",
                     mask_others=False, meta=None):
    """推理复用 motion_eval.evaluate（batch_callback 流式吸收），内存与样本数解耦。"""
    from bevflow.evaluators.motion_eval import evaluate as evaluate_base

    acc = GeneralAccumulator(num_future_sweeps)
    base = evaluate_base(
        model, dataset_root, split=split, out_dir=None, batch_size=batch_size,
        num_workers=num_workers, device=device, use_adj_frame_pred=use_adj_frame_pred,
        use_motion_state_pred_masking=use_motion_state_pred_masking,
        future_frame_skip=future_frame_skip, num_future_sweeps=num_future_sweeps,
        run_id=run_id, ckpt_path=ckpt_path, meta=meta, batch_callback=acc,
    )
    ext = acc.finalize()
    metrics = dict(base)
    metrics["spatial"] = ext["spatial"]
    metrics["stability"] = ext["stability"]
    metrics["meta"] = {**base.get("meta", {}), "eval_protocol": "motionnet_v1_ext"}

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "metrics_general.json"), "w") as f:
            json.dump(metrics, f, indent=1)
    return metrics
