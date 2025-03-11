import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class MotionHungarianMatcher(nn.Module):
    def __init__(self,
                 cost_mask: float = 1.0,
                 cost_class: float = 1.0,
                 cost_disp: float = 1.0,
                 cost_state: float = 1.0):
        super().__init__()
        self.cost_mask = cost_mask
        self.cost_class = cost_class
        self.cost_disp = cost_disp
        self.cost_state = cost_state

    @torch.no_grad()
    def forward(self, preds, gts, non_empty_maps=None):
        """
            non_empty_maps: [bs, h, w]
        """
        pred_masks = preds["masks"]  # [bs, N, h, w]
        pred_classes = preds["classes"]  # [bs, N, num_cls]
        pred_disps = preds["displacements"]  # [bs, seq, H, W, 2]
        pred_static = preds["static"]  # [bs, N, num_state_cls]

        gt_masks = gts["instance_masks"]  # [bs, M, h, w]
        gt_classes = gts["classes"]  # [bs, num_cls, h, w]
        gt_disps = gts["displacements"]  # [bs, seq, h, w, 2]
        gt_static = gts["static"]  # [bs, 2, h, w]

        bs, N = pred_masks.shape[:2]
        M = gt_masks.shape[1]

        C = torch.zeros(bs, N, M, device=pred_masks.device)

        for b in range(bs):
            # Mask代价
            pred_masks_b = pred_masks[b].flatten(1)  # [N, h*w]
            gt_masks_b = gt_masks[b].flatten(1)  # [M, h*w]
            if non_empty_maps is not None:
                non_empty_map_flat = non_empty_maps[b].unsqueeze(0).flatten(start_dim=1)
                pred_flat = pred_flat * non_empty_map_flat
                gt_flat = gt_flat * non_empty_map_flat

            # TODO: add non empty maps
            intersection = pred_masks_b @ gt_masks_b.T  # [N, M]
            union = pred_masks_b.sum(1, keepdim=True) + gt_masks_b.sum(1, keepdim=True).T - intersection
            iou = intersection / (union + 1e-8)
            cost_mask = 1 - iou  # [N, M]

            # 类别代价
            gt_class_ids = gt_classes[b].argmax(0)  # [h, w]
            gt_instance_classes = []
            for m in range(M):
                instance_pixels = gt_masks[b, m] > 0.5
                # breakpoint()
                cls_counts = torch.bincount(gt_class_ids[instance_pixels].flatten())
                dominant_cls = cls_counts.argmax()
                gt_instance_classes.append(dominant_cls)
            gt_instance_classes = torch.stack(gt_instance_classes)  # [M]
            pred_cls_probs = F.softmax(pred_classes[b], -1)  # [N, num_cls]
            cost_class = -pred_cls_probs[:, gt_instance_classes]  # [N, M]

            # 位移代价
            pred_disp_means = []
            for n in range(N):
                mask = pred_masks[b, n]> 0.5
                if mask.sum() == 0:
                    pred_disp_means.append(torch.zeros(seq_len, 2))
                else:
                    pred_disp_means.append(pred_disps[b][:, mask].mean(dim=1))
            pred_disp_means = torch.stack(pred_disp_means)  # [N, seq, 2]
            # breakpoint()

            gt_disp_means = []
            for m in range(M):
                mask = gt_masks[b, m]> 0.5
                gt_disp_means.append(gt_disps[b][:, mask].mean(dim=1))
            gt_disp_means = torch.stack(gt_disp_means)  # [M, seq, 2]

            cost_disp = torch.cdist(pred_disp_means.flatten(1), gt_disp_means.flatten(1), p=1)  # [N, M]

            # 动静状态代价
            # gt_static_labels = gt_static[b].argmax(0).flatten()  # [h*w]
            # cost_static = torch.zeros(N, M, device=pred_masks.device)
            # for n in range(N):
            #     for m in range(M):
            #         instance_area = gt_masks[b, m].flatten() > 0.5
            #         gt_static_label = gt_static_labels[instance_area].mode().values
            #         pred_static_prob = pred_static[b, n]  # [num_state_cls]
            #         cost_static[n, m] = F.cross_entropy(pred_static_prob.unsqueeze(0), gt_static_label.unsqueeze(0))

            # --- 动静状态代价 ---
            gt_state_labels = gt_static[b].argmax(0)  # [h, w]，获取动静标签
            gt_instance_state = []
            cost_state = torch.zeros(N, M, device=pred_masks.device)


            # 获取当前预测mask的区域
            for m in range(M):
                instance_area = gt_masks[b, m] > 0.5  # 选择实例区域
                state_counts = torch.bincount(gt_state_labels[instance_area].flatten())
                dominant_state = state_counts.argmax()
                gt_instance_state.append(dominant_state)
            gt_instance_state = torch.stack(gt_instance_state) # [M]
            pred_state_probs = F.softmax(pred_static[b], -1) # [N, num_state_cls]
            cost_state = -pred_state_probs[:, gt_instance_state]


            # 加权总代价
            C[b] = (self.cost_mask * cost_mask +
                    self.cost_class * cost_class +
                    self.cost_disp * cost_disp +
                    self.cost_state * cost_state)


        # 匈牙利匹配
        indices = [linear_sum_assignment(C[b].cpu()) for b in range(bs)]
        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


if __name__ == "__main__":
    # 示例：创建一些虚拟数据来测试MotionHungarianMatcher
    bs, N, M, h, w = 1, 10, 4, 5, 6
    seq_len = 7
    num_cls = 8
    num_state_cls = 2

    # 创建虚拟预测数据
    pred_masks = torch.rand(bs, N, h, w)
    pred_classes = torch.rand(bs, N, num_cls)
    pred_disps = torch.rand(bs, seq_len, h, w, 2)
    pred_static = torch.rand(bs, N, num_state_cls)  # state

    # 创建虚拟GT数据
    gt_masks = torch.rand(bs, M, h, w)
    gt_classes = torch.rand(bs, num_cls, h, w)
    gt_disps = torch.rand(bs, seq_len, h, w, 2)
    gt_static = torch.rand(bs, num_state_cls, h, w)

    preds = {"masks": pred_masks, "classes": pred_classes, "displacements": pred_disps, "static": pred_static}
    gts = {"instance_masks": gt_masks, "classes": gt_classes, "displacements": gt_disps, "static": gt_static}

    matcher = MotionHungarianMatcher()
    matches = matcher(preds, gts)
    print(matches)
    # output： [(tensor([5, 7, 8, 9]), tensor([3, 2, 1, 0]))]
