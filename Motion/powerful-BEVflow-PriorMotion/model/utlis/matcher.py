import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import numpy as np

def motion_mask_loss(preds, gts, indices, only_mask=True):
    mask_loss_fn = nn.BCEWithLogitsLoss()
    # mask_loss_fn = nn.BCELoss()
    # mask_loss_fn = nn.MSELoss()

    cls_loss_fn = nn.CrossEntropyLoss()
    disp_loss_fn = nn.L1Loss(reduction='none')  # 使用无缩减版本以方便平均

    total_mask_loss = 0
    total_cls_loss = 0
    total_disp_loss = 0

    for b in range(len(indices)):
        src_idx, tgt_idx = indices[b]

        # Mask损失
        pred_masks_b = preds["masks"][b][src_idx]
        gt_masks_b = gts["instance_masks"][b][tgt_idx].float()
        breakpoint()
        total_mask_loss += mask_loss_fn(pred_masks_b, gt_masks_b)

        if not only_mask:
            # 分类损失
            pred_classes_b = preds["classes"][b]
            for i, t in zip(src_idx, tgt_idx):
                instance_area = gts["instance_masks"][b, t] > 0.5  # 实例mask区域
                if instance_area.sum() == 0:
                    continue  # 如果没有像素属于该实例，则跳过
                pred_class_instance = pred_classes_b[:, instance_area].permute(1, 0)  # [num_instance_pixels, num_cls]
                gt_class_instance = gts["classes"][b][:, instance_area].argmax(0)  # [num_instance_pixels]
                total_cls_loss += cls_loss_fn(pred_class_instance, gt_class_instance)

            # 位移损失
            pred_displacements_b = preds["displacements"][b]
            gt_displacements_b = gts["displacements"][b]
            for i, t in zip(src_idx, tgt_idx):
                instance_area = gts["instance_masks"][b, t] > 0.5
                if instance_area.sum() == 0:
                    continue  # 如果没有像素属于该实例，则跳过
                pred_disp_instance = pred_displacements_b[:, instance_area, :]  # [seq_len, num_instance_pixels, 2]
                gt_disp_instance = gt_displacements_b[:, instance_area, :]  # [seq_len, num_instance_pixels, 2]
                loss_disp = disp_loss_fn(pred_disp_instance, gt_disp_instance).mean(dim=[1, 2])  # 对每个序列取平均
                total_disp_loss += loss_disp.mean()  # 对所有序列取平均
    if only_mask:
        return total_mask_loss
    else:
        return {
            "total_loss": total_mask_loss + total_cls_loss + total_disp_loss,
            "mask_loss": total_mask_loss,
            "cls_loss": total_cls_loss,
            "disp_loss": total_disp_loss
        }

class MotionHungarianMatcher(nn.Module):
    def __init__(self,
                 cost_mask: float = 1.0,
                 cost_class: float = 1.0,
                 cost_disp: float = 1.0,
                 cost_state: float = 1.0,
                 cost_binary_class: float = 1.0):  # 添加二值化分类损失
        super().__init__()
        self.cost_mask = cost_mask
        self.cost_class = cost_class
        self.cost_disp = cost_disp
        self.cost_state = cost_state
        self.cost_binary_class = cost_binary_class  # 新的二值化分类损失系数

    @torch.no_grad()
    def forward(self, preds, gts, non_empty_maps=None):
        """
            non_empty_maps: [bs, h, w]
        """
        pred_masks = preds["masks"]  # [bs, N, H, W]
        pred_classes = preds["classes"]  # [bs, num_cls, H, W]
        pred_disps = preds["displacements"]  # [bs, seq, H, W, 2]
        pred_static = preds["static"]  # [bs, 2, H, W]

        gt_masks_list = gts["instance_masks"]  # list of tensors, each of shape [M_i, H, W]
        gt_classes = gts["classes"]  # [bs, num_cls, H, W]
        gt_disps = gts["displacements"]  # [bs, seq, H, W, 2]
        gt_static = gts["static"]  # [bs, 2, H, W]

        bs, N = pred_masks.shape[:2]
        seq_len = pred_disps.shape[1]

        indices = []
        for b in range(bs):
            M = gt_masks_list[b].shape[0]  # 当前样本的有效实例数

            # --- Mask代价 ---
            pred_masks_b = pred_masks[b].flatten(1)  # [N, H*W]
            gt_masks_b = gt_masks_list[b].flatten(1).float()  # [M, H*W]
            if non_empty_maps is not None:
                non_empty_map_flat = non_empty_maps[b].unsqueeze(0).flatten(start_dim=1)
                pred_masks_b = pred_masks_b * non_empty_map_flat
                gt_masks_b = gt_masks_b * non_empty_map_flat

            intersection = pred_masks_b @ gt_masks_b.T  # [N, M]
            union = pred_masks_b.sum(1, keepdim=True) + gt_masks_b.sum(1, keepdim=True).T - intersection
            iou = intersection / (union + 1e-8)
            cost_mask = 1 - iou  # [N, M]

            # --- 类别代价 ---
            pred_cls_probs = F.softmax(pred_classes[b], dim=0)  # [num_cls, H, W]
            cost_class = torch.zeros(N, M, device=pred_masks.device)
            for m in range(M):
                instance_area = gt_masks_list[b][m] > 0  # [H, W]
                if instance_area.sum() == 0:
                    continue
                pred_cls_probs_instance = pred_cls_probs[:, instance_area]  # [num_cls, num_instance_pixels]
                gt_cls_probs_instance = gt_classes[b][:, instance_area]  # [num_cls, num_instance_pixels]
                cost_class += F.cross_entropy(pred_cls_probs_instance.permute(1, 0), gt_cls_probs_instance.argmax(0)).unsqueeze(0).repeat(N, 1)

            # --- 位移代价 ---
            last_pred_disp = pred_disps[b, -1]  # [H, W, 2]
            last_gt_disp = gt_disps[b, -1]  # [H, W, 2]
            pred_disp_means = []
            gt_disp_means = []

            for n in range(N):
                mask = pred_masks[b, n] > 0.5  # [H, W]
                if mask.sum() == 0:
                    pred_disp_means.append(torch.zeros(2, device=pred_masks.device))
                else:
                    pred_disp_means.append(last_pred_disp[mask].mean(dim=0))  # [2]
            pred_disp_means = torch.stack(pred_disp_means)  # [N, 2]

            for m in range(M):
                mask = gt_masks_list[b][m] > 0.5
                if mask.sum() == 0:
                    gt_disp_means.append(torch.zeros(2, device=pred_masks.device))
                else:
                    gt_disp_means.append(last_gt_disp[mask].mean(dim=0))  # [2]
            try:
                gt_disp_means = torch.stack(gt_disp_means)  # [M, 2]
            except:
                breakpoint()

            cost_disp = torch.cdist(pred_disp_means, gt_disp_means, p=1)  # [N, M]

            # --- 动静状态代价 ---
            pred_state_probs = F.softmax(pred_static[b], dim=0)  # [2, H, W]
            cost_state = torch.zeros(N, M, device=pred_masks.device)
            for m in range(M):
                instance_area = gt_masks_list[b][m] > 0  # [H, W]
                if instance_area.sum() == 0:
                    continue
                pred_state_cls_probs_instance = pred_state_probs[:, instance_area]  # [2, num_instance_pixels]
                gt_state_probs_instance = gt_static[b][:, instance_area]  # [2, num_instance_pixels]
                cost_state += F.cross_entropy(pred_state_cls_probs_instance.permute(1, 0), gt_state_probs_instance.argmax(0)).unsqueeze(0).repeat(N, 1)

            # 加权总代价
            C = (self.cost_mask * cost_mask +
                 self.cost_class * cost_class +
                 self.cost_disp * cost_disp +
                 self.cost_state * cost_state)

            # 匈牙利匹配
            row_ind, col_ind = linear_sum_assignment(C.detach().cpu().numpy())
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64),
                            torch.as_tensor(col_ind, dtype=torch.int64)))

        return indices


# 测试功能
if __name__ == "__main__":
    batch_size = 2
    num_predictions = 20
    height, width = 4, 4
    num_classes = 5
    sequence_length = 10
    threshold = 0.5
    # 模拟预测结果preds
    prediction_masks = torch.rand(batch_size, num_predictions, height, width)  # [bs, N, h, w]
    # binary mask
    # prediction_masks = (prediction_masks > threshold).float()
    # prediction_masks = prediction_masks.sigmoid()
    prediction_masks = prediction_masks.softmax(dim=1)
    prediction_classes = torch.rand(batch_size, num_classes, height, width)  # [bs, num_cls, H, W]
    prediction_displacements = torch.rand(batch_size, sequence_length, height, width, 2)  # [bs, seq, H, W, 2]
    prediction_static = torch.rand(batch_size, 2, height, width)  # [bs, 2, H, W]

    preds = {
        "masks": prediction_masks,
        "classes": prediction_classes,
        "displacements": prediction_displacements,
        "static": prediction_static,
    }

    # 对 N 这个维度进行 shuffle
    shuffled_prediction_masks = torch.zeros_like(prediction_masks)  # 创建一个相同大小的零张量来存储shuffle后的结果

    for i in range(batch_size):
        # 为每个batch item生成一个随机排列的索引
        idx = torch.randperm(num_predictions)

        # 使用上述随机索引对当前batch item的'N'维度进行shuffle
        shuffled_prediction_masks[i] = prediction_masks[i][idx]

    # 模拟真实值gts
    # ground_truth_instance_masks_list = [
    #     torch.rand(np.random.randint(1, 4), height, width) > 0.5 for _ in range(batch_size)
    # ]  # 每个样本的实例mask数量随机，且为布尔型
    ground_truth_instance_masks_list = [shuffled_prediction_masks[bs] for bs in range(0, shuffled_prediction_masks.shape[0])]
    print(ground_truth_instance_masks_list[0].shape, ground_truth_instance_masks_list[1].shape)
    # ground_truth_instance_masks_list = prediction_masks
    ground_truth_classes = torch.rand(batch_size, num_classes, height, width)  # [bs, num_cls, h, w]
    ground_truth_displacements = torch.rand(batch_size, sequence_length, height, width, 2)  # [bs, seq, H, W, 2]
    ground_truth_static = torch.rand(batch_size, 2, height, width)  # [bs, 2, h, w]

    gts = {
        "instance_masks": ground_truth_instance_masks_list,
        "classes": ground_truth_classes,
        "displacements": ground_truth_displacements,
        "static": ground_truth_static,
    }

    matcher = MotionHungarianMatcher()
    indices = matcher(preds, gts)
    print(indices)

    mask_loss = motion_mask_loss(preds, gts, indices, only_mask=True)
    print(mask_loss)
