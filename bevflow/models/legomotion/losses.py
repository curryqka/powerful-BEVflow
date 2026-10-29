# SPDX-License-Identifier: Apache-2.0
# 注意：本文件在运行时依赖 AGPL-3.0-or-later 许可的 bevflow.models.motionnet 原语
# （STPN/三头等，派生自 MERL MotionNet）。组合使用时的许可见 LICENSE。
"""LEGO-Motion 掩码损失（spec 式10 + 用户记忆口径）。

匹配代价 C = 1.0·(−p_cls) + 5.0·BCE + 5.0·Dice；no-object 类 0.1 降权；
匹配后 L_mask-cls = CE(类别) + 5.0·BCE + 5.0·Dice。
两阶段 λ 调度由 trainer 注入（前 20 epoch mask=2.0；后 25 epoch 网格项=2.0）。
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

CAT_NUM = 5
NUM_CLASSES = CAT_NUM + 1  # 含 no-object


def build_mask_targets(instance_map, pixel_cat):
    """instance_map (B,H,W) uint8；pixel_cat (B,H,W,5) → 每样本 (masks (Ng,H,W) bool, classes (Ng,))。

    类别取实例内格子 argmax 后的众数。
    """
    out = []
    for b in range(instance_map.shape[0]):
        inst = np.asarray(instance_map[b])
        ids = np.unique(inst)
        ids = ids[ids != 0]
        masks, classes = [], []
        for iid in ids:
            m = inst == iid
            masks.append(m)
            cells = np.argwhere(m)
            cats = np.asarray(pixel_cat[b])[cells[:, 0], cells[:, 1]].argmax(-1)
            vals, counts = np.unique(cats, return_counts=True)
            classes.append(int(vals[counts.argmax()]))
        out.append((
            np.stack(masks) if masks else np.zeros((0, *inst.shape), dtype=bool),
            np.asarray(classes, dtype=np.int64),
        ))
    return out


def _sigmoid_focal_free_bce(logits, target):
    """无 reduction 的 BCE，逐对平均（匹配代价用）。"""
    return -(target * F.logsigmoid(logits) + (1 - target) * F.logsigmoid(-logits))


def _dice(prob, target, eps=1.0):
    num = 2 * (prob * target).sum(-1) + eps
    den = prob.sum(-1) + target.sum(-1) + eps
    return 1 - num / den


def match_batch(pred_masks, pred_logits, gt_targets, w_cls=1.0, w_bce=5.0, w_dice=5.0):
    """逐样本匈牙利匹配。pred_masks (B,N,HW) logits；pred_logits (B,N,K+1)。"""
    log_probs = pred_logits.log_softmax(-1)
    matches = []
    for b in range(pred_masks.shape[0]):
        gt_masks, gt_classes = gt_targets[b]
        if len(gt_classes) == 0:
            matches.append((np.array([], dtype=int), np.array([], dtype=int)))
            continue
        tgt = torch.as_tensor(gt_masks, dtype=pred_masks.dtype, device=pred_masks.device).flatten(1)
        # 逐 GT 实例计算代价，避免 N×Ng×HW 的广播中间量（峰值 ~4GB → ~67MB）
        n_gt = tgt.shape[0]
        bce = torch.empty((pred_masks.shape[1], n_gt), dtype=pred_masks.dtype, device=pred_masks.device)
        dice = torch.empty_like(bce)
        prob = pred_masks[b].sigmoid()
        with torch.no_grad():
            for j in range(n_gt):
                bce[:, j] = _sigmoid_focal_free_bce(pred_masks[b], tgt[j]).mean(-1)
                dice[:, j] = _dice(prob, tgt[j])
            cls_cost = -log_probs[b][:, torch.as_tensor(gt_classes, device=pred_masks.device)]
            cost = w_cls * cls_cost + w_bce * bce + w_dice * dice
        ri, ci = linear_sum_assignment(cost.detach().cpu().numpy())
        matches.append((ri, ci))
    return matches


def mask_cls_loss(pred_masks, pred_logits, gt_targets, matches, no_obj_weight=0.1):
    """返回 (loss, parts)。对预测做平均（MaskFormer 口径）。"""
    ce_terms, bce_terms, dice_terms = [], [], []
    log_probs = pred_logits.log_softmax(-1)
    for b in range(pred_masks.shape[0]):
        ri, ci = matches[b]
        n_pred = pred_masks.shape[1]
        matched_pred = set(ri.tolist())
        gt_masks, gt_classes = gt_targets[b]

        cls_ce = -log_probs[b][:, NUM_CLASSES - 1] * no_obj_weight  # 未匹配默认 no-object（CE 为正）
        if len(ri):
            gt_cls = torch.as_tensor(gt_classes[ci], device=pred_logits.device)
            cls_ce = cls_ce.index_put((torch.as_tensor(ri, device=pred_logits.device),), -log_probs[b][
                torch.as_tensor(ri, device=pred_logits.device), gt_cls], accumulate=False)
            tgt = torch.as_tensor(gt_masks[ci], dtype=pred_masks.dtype, device=pred_masks.device).flatten(1)
            sel = pred_masks[b][torch.as_tensor(ri, device=pred_masks.device)]
            bce = _sigmoid_focal_free_bce(sel, tgt).mean(-1)
            dice = _dice(sel.sigmoid(), tgt)
            bce_terms.append(bce.mean())
            dice_terms.append(dice.mean())
        ce_terms.append(cls_ce.mean())
    ce = torch.stack(ce_terms).mean()
    bce = torch.stack(bce_terms).mean() if bce_terms else torch.zeros((), device=pred_masks.device)
    dice = torch.stack(dice_terms).mean() if dice_terms else torch.zeros((), device=pred_masks.device)
    loss = ce + 5.0 * bce + 5.0 * dice
    return loss, {"mask_ce": float(ce), "mask_bce": float(bce), "mask_dice": float(dice)}


def lambda_schedule(epoch, mask_weight_early=2.0, grid_weight_late=2.0, switch_epoch=20):
    """两阶段调度（论文 §IV.A）：前 20 epoch grid=1.0/mask=2.0；之后 grid=2.0/mask=1.0。"""
    if epoch <= switch_epoch:
        return {"grid": 1.0, "mask": mask_weight_early}
    return {"grid": grid_weight_late, "mask": 1.0}
