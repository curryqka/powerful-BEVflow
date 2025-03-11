import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
# import torch
# import torch.nn.functional as F
# from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
class InstanceSegmentationLoss(nn.Module):
    def __init__(self, matcher, weight_dict=None):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict or {
            'loss_mask': 1.0,
            'loss_dice': 0.05,
            'loss_focal': 1.0,
            'loss_ce': 1.0,
            'loss_displace': 1.0,
            'loss_state': 1.0,
        }

    def _get_matched_indices(self, outputs, targets, valid_count):
        # 这里假设matcher返回每个batch的匹配索引
        # 具体实现需根据实际匹配逻辑调整

        return self.matcher(outputs, targets, valid_count)

    def forward(self, pred_masks, pred_classes, pred_displacements, pred_states,
                gt_instance_masks, gt_classes, gt_displacements, gt_states,
                valid_count, non_empty_map=None, disp_valid_map=None):
        """
        non_empty_map: [bs, H, W]
        disp_valid_map: [bs, seq, H, W, 1]
        """

        preds = {"masks": pred_masks, "classes": pred_classes, "displacements": pred_displacements, "static": pred_states}
        gts = {"instance_masks": gt_instance_masks, "classes": gt_classes, "displacements": gt_displacements, "static": gt_states}
        # 获取匹配结果，indices是列表，每个元素为(batch_idx, src_idx, tgt_idx)

        indices = self._get_matched_indices(
            outputs=preds,
            targets=gts,
            valid_count=valid_count,
        )
        # breakpoint()

        # boardcast non_empty_map
        non_empty_map_copy = non_empty_map.clone()[:, None, ...].repeat(1, pred_masks.shape[1], 1, 1)
        # breakpoint()
        # 收集匹配后的预测和GT
        src_masks, tgt_masks = [], []
        src_classes, tgt_classes = [], []
        src_disps, tgt_disps = [], []
        src_states, tgt_states = [], []
        src_non_empty, tgt_non_empty = [], []

        for b in range(0, len(indices)):
            # 提取当前batch的匹配对
            src_idx, tgt_idx = indices[b]
            for src, tgt in zip(src_idx, tgt_idx):
                src_masks.append(pred_masks[b, src])
                tgt_masks.append(gt_instance_masks[b, tgt])
                src_classes.append(pred_classes[b, src])
                tgt_classes.append(gt_classes[b, tgt])
                src_disps.append(pred_displacements[b, src])
                tgt_disps.append(gt_displacements[b, tgt])

                src_states.append(pred_states[b, src])
                tgt_states.append(gt_states[b, tgt])
                src_non_empty.append(non_empty_map_copy[b, src])
                tgt_non_empty.append(non_empty_map_copy[b, src])

        if not src_masks:
            return {'loss': torch.tensor(0., device=pred_masks.device)}

        # 拼接所有匹配对
        try:
            src_masks = torch.stack(src_masks, dim=0)
        except:
            breakpoint()
        tgt_masks = torch.stack(tgt_masks, dim=0)
        src_classes = torch.stack(src_classes, dim=0)
        tgt_classes = torch.stack(tgt_classes, dim=0)
        src_disps = torch.stack(src_disps, dim=0)
        tgt_disps = torch.stack(tgt_disps, dim=0)

        src_states = torch.stack(src_states, dim=0)
        tgt_states = torch.stack(tgt_states, dim=0)
        src_non_empty = torch.stack(src_non_empty, dim=0)
        tgt_non_empty = torch.stack(tgt_non_empty, dim=0)
        num_masks = src_masks.shape[0]
        device = tgt_non_empty.device
        # 计算mask损失
        dice = dice_loss(src_masks, tgt_masks, num_masks, non_empty_map=tgt_non_empty)
        focal = sigmoid_focal_loss(src_masks, tgt_masks, num_masks, non_empty_map=non_empty_map)
        mask_loss = self.weight_dict['loss_dice'] * dice + self.weight_dict['loss_focal'] * focal

        # 计算类别损失（假设GT是one-hot编码）
        # breakpoint()
        # TODO: class weighted
        cls_weight = [0.005, 1.0, 1.0, 1.0, 1.0]

        tgt_cls_labels = torch.argmax(tgt_classes, dim=1) + 1
        class_weights = torch.zeros_like(tgt_cls_labels)
        # breakpoint()
        for k in range(0, len(cls_weight)):
            mask = tgt_cls_labels == (k+1)
            class_weights[mask] = cls_weight[k]

        # class_weights = weight[tgt_cls_labels.long()]
        # ce_loss = F.cross_entropy(src_classes, tgt_cls_labels, weight=torch.tensor(weight).to(device))
        log_softmax_probs = F.log_softmax(src_classes, dim=1)
        ce_loss = torch.sum(-tgt_classes * log_softmax_probs, dim=1) * class_weights

        # 计算位移损失
        # TODO: disp_valid_map: [bs, seq, h, w, 1]

        displace_loss = F.mse_loss(src_disps, tgt_disps, reduction='none').mean(dim=(1,2))
        breakpoint()
        displace_loss = class_weights * displace_loss
        displace_loss = displace_loss.mean()

        # calculate state loss
        # TODO: class weighted
        state_weight = [0.005, 1.0]
        tgt_state_labels = torch.argmax(tgt_states, dim=1) + 1
        state_weights = torch.zeros_like(tgt_state_labels)
        for k in range(0, len(state_weight)):
            mask = tgt_state_labels == (k+1)
            state_weights[mask] = state_weight[k]


        # state_loss = F.cross_entropy(src_states, tgt_state_labels, weight=torch.tensor(state_weight).to(device))
        log_softmax_probs = F.log_softmax(src_states, dim=1)
        state_loss = torch.sum(-tgt_states * log_softmax_probs, dim=1) * state_weights
        # 加权总损失
        losses = {
            'loss_mask': mask_loss * self.weight_dict['loss_mask'],
            # 'loss_dice': dice * self.weight_dict['loss_dice'],
            'loss_ce': ce_loss * self.weight_dict['loss_ce'],
            'loss_displace': displace_loss * self.weight_dict['loss_displace'],
            'loss_state': state_loss * self.weight_dict['loss_state']
        }
        losses['loss'] = sum(losses.values())

        return losses

def dice_loss(inputs, targets, num_masks, non_empty_map=None):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: Number of masks used to normalize the loss.
    """
    # breakpoint()
    inputs = inputs
    inputs = inputs.flatten(1)
    targets = targets.flatten(1).float()
    non_empty_map = non_empty_map.flatten(1).float()
    # breakpoint()
    if non_empty_map is not None:
        inputs = inputs * non_empty_map
        targets = targets * non_empty_map

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2, non_empty_map=None):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        num_masks: Number of masks used to normalize the loss.
        alpha: Weighting factor for class balancing.
        gamma: Exponent for modulating the loss.
    """
    # prob = inputs.sigmoid()
    prob = inputs
    targets = targets.float()
    # breakpoint()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if non_empty_map is not None:
        loss = loss * non_empty_map
    return loss.mean(1).sum() / num_masks
def dummy_matcher(outputs, targets, valid_count):
    # 简单返回每个batch中的第一个预测与GT之间的匹配
    return [(i, [0], [0]) for i in range(len(outputs[0]))]

class MotionHungarianMatcher(torch.nn.Module):
    def __init__(self,
                 cost_mask: float = 1.0,
                 cost_class: float = 1.0,
                 cost_disp: float = 1.0,
                 cost_state: float = 1.0,
                 focal_alpha: float = 0.25):
        super().__init__()
        self.cost_mask = cost_mask
        self.cost_class = cost_class
        self.cost_disp = cost_disp
        self.cost_state = cost_state
        self.focal_alpha = focal_alpha

    @staticmethod
    def batch_dice_loss(inputs, targets):
        # inputs = inputs.sigmoid()
        numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
        return 1 - (numerator + 1) / (denominator + 1)

    def batch_sigmoid_focal_loss(self, inputs, targets):
        hw = inputs.shape[1]
        # prob = inputs.sigmoid()
        prob = inputs
        focal_pos = ((1 - prob)**2) * F.binary_cross_entropy_with_logits(
            inputs, torch.ones_like(inputs), reduction='none'
        )
        focal_neg = (prob**2) * F.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(inputs), reduction='none'
        )
        loss = (
            torch.einsum("nc,mc->nm", focal_pos, targets) * self.focal_alpha +
            torch.einsum("nc,mc->nm", focal_neg, (1 - targets)) * (1 - self.focal_alpha)
        )
        return loss / hw

    @torch.no_grad()
    def forward(self, preds, gts, valid_counts, non_empty_map=None):
        """
        non_empty_map: [bs, H, W]
        """
        pred_masks = preds["masks"]  # [bs, N, H, W]
        pred_classes = preds["classes"]  # [bs, N, num_cls]
        pred_disps = preds["displacements"]  # [bs, N, seq_len, 2]
        pred_states = preds["static"]  # [bs, N, num_state_cls]

        gt_masks = gts["instance_masks"]  # [bs, M, H, W]
        gt_classes = gts["classes"]  # [bs, M, num_cls]
        gt_disps = gts["displacements"]  # [bs, M, seq_len, 2]
        gt_states = gts["static"]  # [bs, M, num_state_cls]

        bs, N = pred_masks.shape[:2]
        device = pred_masks.device

        indices = []

        for b in range(bs):
            M = valid_counts[b].item()

            # Mask代价计算
            pred_flat = pred_masks[b][:N].flatten(start_dim=1)  # [N, H*W]
            gt_flat = gt_masks[b][:M].flatten(start_dim=1)  # [M, H*W]
            if non_empty_map is not None:
                non_empty_map_flat = non_empty_map[b].unsqueeze(0).flatten(start_dim=1)
                pred_flat = pred_flat * non_empty_map_flat
                gt_flat = gt_flat * non_empty_map_flat

            cost_focal = self.batch_sigmoid_focal_loss(pred_flat, gt_flat.float())
            cost_dice = self.batch_dice_loss(pred_flat, gt_flat.float())
            mask_cost = cost_focal + cost_dice

            # 类别代价计算
            pred_cls_prob = F.softmax(pred_classes[b], dim=-1)  # [N, num_cls]
            gt_class_ids = gt_classes[b][:M].argmax(dim=-1)  # [M]
            class_cost = -pred_cls_prob[:, gt_class_ids]  # [N, M]

            # 位移代价计算
            pred_disp_mean = pred_disps[b][:N].mean(dim=1)  # [N, seq_len, 2]
            gt_disp_mean = gt_disps[b][:M].mean(dim=1)  # [M, seq_len, 2]
            # disp_cost = torch.cdist(pred_disp_mean.flatten(start_dim=1),
            #                         gt_disp_mean.flatten(start_dim=1), p=1)
            disp_cost = torch.cdist(pred_disp_mean, gt_disp_mean, p=1)  # [N, M]

            # 状态代价计算
            pred_state_prob = F.softmax(pred_states[b], dim=-1)  # [N, num_state_cls]
            gt_state_ids = gt_states[b][:M].argmax(dim=-1)  # [M]
            state_cost = -pred_state_prob[:, gt_state_ids]  # [N, M]

            # 总代价计算
            C = (self.cost_mask * mask_cost +
                 self.cost_class * class_cost +
                 self.cost_disp * disp_cost +
                 self.cost_state * state_cost)


            # 匈牙利匹配
            row_ind, col_ind = linear_sum_assignment(C.detach().cpu().numpy())
            # row_tensor = torch.as_tensor(row_ind, dtype=torch.int64, device=device)
            # col_tensor = torch.as_tensor(col_ind, dtype=torch.int64, device=device)

            indices.append((torch.as_tensor(row_ind, dtype=torch.int64, device=device),
                            torch.as_tensor(col_ind, dtype=torch.int64, device=device)))
            # 格式化输出为所需的格式
            # indices.append((b, row_tensor.unsqueeze(0), col_tensor.unsqueeze(0)))

        return indices


# 示例用法略...
if __name__ == "__main__":
    # 示例：创建一些虚拟数据来测试MotionHungarianMatcher
    bs, N, M, h, w = 32, 10, 4, 2, 2
    seq_len = 7
    num_cls = 8
    num_state_cls = 2

    # 创建虚拟预测数据
    pred_masks = torch.rand(bs, N, h, w)
    pred_masks = pred_masks.sigmoid()
    pred_masks = (pred_masks > 0.5).float()
    pred_classes = torch.rand(bs, N, num_cls)
    pred_disps = torch.rand(bs, N, seq_len, 2)
    pred_static = torch.rand(bs, N, num_state_cls)  # state

    # 创建虚拟GT数据
    # 对 N 这个维度进行 shuffle
    shuffled_prediction_masks = torch.zeros_like(pred_masks)  # 创建一个相同大小的零张量来存储shuffle后的结果
    gt_masks_list = [shuffled_prediction_masks[bs] for bs in range(0, shuffled_prediction_masks.shape[0])]

    gt_masks_list = pred_masks
    # gt_masks = torch.rand(bs, M, h, w)
    # gt_masks = torch.stack(gt_masks_list, dim=0)
    gt_classes = torch.rand(bs, M, num_cls)
    gt_disps = torch.rand(bs, M, seq_len, 2)
    gt_static = torch.rand(bs, M, num_state_cls)
    # valid_counts = torch.randint()
    valid_counts = torch.randint(low=1, high=M+1, size=(bs,))
    non_empty_map = torch.rand(bs, h, w)
    disp_valid_map = torch.rand(bs, seq_len, h, w, 1)
    print("valid counts per-batch: {}".format(valid_counts))
    preds = {"masks": pred_masks, "classes": pred_classes, "displacements": pred_disps, "static": pred_static}
    gts = {"instance_masks": gt_masks_list, "classes": gt_classes, "displacements": gt_disps, "static": gt_static}

    matcher = MotionHungarianMatcher(cost_mask=1.0,
                 cost_class= 0,
                 cost_disp= 0,
                 cost_state = 0,
                 focal_alpha= 0.25)
    matches = matcher(preds, gts, valid_counts)
    print(matches)

    losser = InstanceSegmentationLoss(matcher=matcher)
    loss_dict = losser(pred_masks, pred_classes, pred_disps, pred_static,
                       gt_masks_list, gt_classes, gt_disps, gt_static,
                       valid_counts, non_empty_map=non_empty_map, disp_valid_map=disp_valid_map)
    print(loss_dict)

    # test loss function
    pred_tensor = torch.rand(N, h, w).sigmoid()
    pred_tensor_binary = (pred_tensor > 0.5).float()
    gt_tensor = pred_tensor
    # gt_tensor = torch.rand(N, h, w)
    # gt_tensor = (gt_tensor > 0.5).float()

    """test dice loss"""
    # dice_test = dice_loss(pred_tensor_binary, gt_tensor, N, non_empty_map=non_empty_map)
    # print(dice_test)
