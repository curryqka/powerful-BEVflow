import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from model.utlis.matcher_final import MotionHungarianMatcher
def choose_disp_weighted_map(gt, map_shape, device, mode = 'motion_status'):

    gt_clone = gt.clone()
    gt_clone[gt < 0] = 0
    gt_clone = gt_clone.numpy()
    disp_weight_map = gt_clone

    bs, _, _, bev_w, bev_h = map_shape
    # [320, 2, 256, 256]
    disp_weight_map = disp_weight_map.reshape(map_shape[0], -1, map_shape[-3], map_shape[-2], map_shape[-1])
    # disp_weight_map = np.mean(disp_weight_map, axis=1).unsqueeze(1)

    if mode == 'norm1':
        disp_weight_map = F.normalize(disp_weight_map, dim=[-2, -1], p=1) * 5
    elif mode == 'norm2':
        disp_weight_map = F.normalize(disp_weight_map, dim=[-2, -1], p=2) * 5
    elif mode == 'softmax':
        disp_weight_map = F.softmax(disp_weight_map, dim=-2)
    elif mode == 'cls':
        # [bs, 1 , 1]
        max_disp_map = disp_weight_map.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)

        # print(max_disp_map.shape)
        # print(disp_weight_map.max(dim=-1)[0].shape)

        # print(max_disp_map[0])
        disp_weight_map[disp_weight_map > 0.5 * max_disp_map] = 2
        # disp_weight_map[torch.where(disp_weight_map > 0.5 * max_disp_map, disp_weight_map, torch.zeros_like(disp_weight_map))] = 2
    elif mode == 'motion_status':
        last_frame_disp_norm = np.linalg.norm(gt_clone, ord=2, axis=-1)

        # [bs, 256, 256]
        # last_frame_disp_norm = last_frame_disp_norm[:, -1, :, :]
        upper_thresh = 0.2
        frame_skip = 3
        upper_bound = (frame_skip + 1) / 20 * upper_thresh
        selected_future_sweeps = np.arange(0, 20 + 1, frame_skip + 1)
        selected_future_sweeps = selected_future_sweeps[1:]

        future_sweeps_disp_field_gt_norm = last_frame_disp_norm[:, -len(selected_future_sweeps):, ...]
        static_cell_mask = future_sweeps_disp_field_gt_norm <= upper_bound
        static_cell_mask = np.all(static_cell_mask, axis=1)  # along the sequence axis
        moving_cell_mask = np.logical_not(static_cell_mask)

        # speed_intervals = np.array([[0, 5.0], [5.0, 20.0], [20.0, np.inf]])
        speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0], [20.0, np.inf]])
        # Next, compute the speed level mask
        last_future_sweep_id = selected_future_sweeps[-1]
        distance_intervals = speed_intervals * (last_future_sweep_id / 20.0)


        last_frame_disp_norm = last_frame_disp_norm[:, -1, :, :]

        weight_vector = [0.95, 1.25, 1.35, 0.75]
        disp_weight_map = np.zeros((bs, bev_w, bev_h), dtype=np.float32)

        for weight, s, d in zip(weight_vector, range(0, speed_intervals.shape[0]), distance_intervals):
            if s == 0:
                mask = static_cell_mask
            else:
                mask = np.logical_and(d[0] <= last_frame_disp_norm, last_frame_disp_norm < d[1])
                mask = np.logical_and(mask, moving_cell_mask)
            disp_weight_map[mask] = weight

        disp_weight_map = disp_weight_map[:, np.newaxis, np.newaxis, ...]
        disp_weight_map = torch.from_numpy(disp_weight_map).to(device)
    elif mode == "motion_status_withoutS":
        last_frame_disp_norm = np.linalg.norm(gt_clone, ord=2, axis=-1)

        # [bs, 256, 256]
        last_frame_disp_norm = last_frame_disp_norm[:, -1, :, :]

        speed_intervals = np.array([[0, 5.0], [5.0, 20.0], [20.0, np.inf]])
        # speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0], [20.0, np.inf]])
        # Next, compute the speed level mask
        # last_future_sweep_id = selected_future_sweeps[-1]
        distance_intervals = speed_intervals * (20.0 / 20.0)

        weight_vector = [1.2, 0.9, 0.4]
        disp_weight_map = np.zeros((bs, bev_w, bev_h), dtype=np.float32)

        for weight, s, d in zip(weight_vector, range(0, speed_intervals.shape[0]), distance_intervals):

            mask = np.logical_and(d[0] <= last_frame_disp_norm, last_frame_disp_norm < d[1])
            disp_weight_map[mask] = weight

        disp_weight_map = disp_weight_map[:, np.newaxis, np.newaxis, ...]
        disp_weight_map = torch.from_numpy(disp_weight_map).to(device)
    else:
        AssertionError,'wrong loss name'
    # loss_disp = torch.sum(loss_disp * cat_weight_map * disp_weight_map) / valid_pixel_num
    return disp_weight_map

def motion_mask_loss(preds, gts, indices, non_empty_maps=None, only_mask=True):
    """
        non_empty_maps: [bs, h, w]
    """
    # mask_loss_fn = nn.BCELoss()
    mask_loss_fn = nn.MSELoss()
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
        # breakpoint()
        if non_empty_maps is not None:
            pred_masks_b = pred_masks_b * non_empty_maps[b]
            gt_masks_b = gt_masks_b * non_empty_maps[b]

            non_empty_maps_copy = non_empty_maps.clone().unsqueeze(1)
        # TODO: add non empty map
        num_masks = src_idx.shape[0]
        dice = dice_loss(pred_masks_b, gt_masks_b, num_masks, non_empty_map=non_empty_maps_copy[b])
        focal = sigmoid_focal_loss(pred_masks_b, gt_masks_b, num_masks, non_empty_map=non_empty_maps_copy[b])
        mask_loss = 0.5 * dice + 0.005 * focal
        total_mask_loss = total_mask_loss + 1 * mask_loss_fn(pred_masks_b, gt_masks_b) + 0 * mask_loss

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
    if non_empty_map is not None:
        non_empty_map = non_empty_map.flatten(1).float()
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
    # breakpoint()
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    # breakpoint()
    if non_empty_map is not None:
        loss = loss * non_empty_map
    # breakpoint()
    return loss.mean(dim=(1,2)).sum() / num_masks

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
        # breakpoint()
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
        focal = sigmoid_focal_loss(src_masks, tgt_masks, num_masks, non_empty_map=tgt_non_empty)
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
        # breakpoint()
        # print(class_weights)
        # class_weights = weight[tgt_cls_labels.long()]
        # ce_loss = F.cross_entropy(src_classes, tgt_cls_labels, weight=torch.tensor(weight).to(device))
        log_softmax_probs = F.log_softmax(src_classes, dim=1)
        ce_loss = torch.sum(-tgt_classes * log_softmax_probs, dim=1) * class_weights
        ce_loss = torch.sum(ce_loss) / src_classes.shape[0]
        # 计算位移损失
        # TODO: disp_valid_map: [bs, seq, h, w, 1]

        displace_loss = F.mse_loss(src_disps, tgt_disps, reduction='none').mean(dim=(1,2))
        # breakpoint()
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
        state_loss = torch.sum(state_loss) / src_states.shape[0]
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

def dummy_matcher(outputs, targets, valid_count, batch_size=2):
    # 简单返回每个batch中的第一个预测与GT之间的匹配
    return [([0], [0]) for i in range(batch_size)]


if __name__ == '__main__':

    # test choose weighted loss
    # map_shape = [16, 1, 1, 256, 256]
    # gt = torch.rand(16, 20, 256, 256, 2)
    # disp_weight_map = choose_disp_weighted_map(gt, map_shape, 'motion_status')
    # 设置随机种子以保证结果可复现
    torch.manual_seed(0)

    batch_size = 2
    num_predictions = 5
    height, width = 4, 4
    num_classes = 3
    sequence_length = 2

    # 模拟预测数据
    prediction_masks = torch.rand(batch_size, num_predictions, height, width)  # [bs, N, h, w]
    prediction_classes = torch.rand(batch_size, num_classes, height, width)  # [bs, num_cls, H, W]
    prediction_displacements = torch.rand(batch_size, sequence_length, height, width, 2)  # [bs, seq, H, W, 2]
    prediction_static = torch.rand(batch_size, 2, height, width)  # [bs, 2, H, W]

    preds = {
        "masks": prediction_masks,
        "classes": prediction_classes,
        "displacements": prediction_displacements,
        "static": prediction_static,
    }
    non_empty_map = torch.rand(batch_size, height, width)
    # 模拟真实标签数据
    ground_truth_instance_masks = (torch.rand(batch_size, num_predictions, height, width) > 0.5).float()  # [bs, N, h, w]
    ground_truth_classes = torch.rand(batch_size, num_classes, height, width)  # [bs, num_cls, h, w]
    ground_truth_displacements = torch.rand(batch_size, sequence_length, height, width, 2)  # [bs, seq, H, W, 2]
    ground_truth_static = torch.eye(2)[torch.randint(0, 2, (batch_size, height, width))].permute(0, 3, 1, 2)  # [bs, 2, h, w]

    gts = {
        "instance_masks": ground_truth_instance_masks,
        "classes": ground_truth_classes,
        "displacements": ground_truth_displacements,
        "static": ground_truth_static,
    }

    # 假设匹配索引（这里我们简单地使用对角线矩阵模拟）
    indices = [(torch.tensor([i for i in range(num_predictions)]), torch.tensor([i for i in range(num_predictions)])) for _ in range(batch_size)]
    indices = [(torch.tensor([0, 1, 2, 4, 3]), torch.tensor([1, 0, 2, 4, 3])) for _ in range(batch_size)]
    # 计算损失
    loss_dict = motion_mask_loss(preds, gts, indices, non_empty_map)
    print(loss_dict)

    # output: {'total_loss': tensor(16.2071), 'mask_loss': tensor(1.4733), 'cls_loss': tensor(11.1202), 'disp_loss': tensor(3.6135)}
    # output: tensor(1.4733)


    batch_size = 2
    num_masks = 3
    height, width = 64, 64
    num_classes, num_state_classes = 5, 2
    seq_len = 20

    # 创建随机预测和真实值
    pred_masks = torch.rand(batch_size, num_masks, height, width)
    pred_classes = torch.randn(batch_size, num_masks, num_classes)
    pred_displacements = torch.randn(batch_size, num_masks, seq_len, 2)
    pred_states = torch.randn(batch_size, num_masks, num_state_classes)

    gt_instance_masks = torch.randint(0, 2, (batch_size, num_masks, height, width)).float()
    gt_classes = F.one_hot(torch.randint(0, num_classes, (batch_size, num_masks)), num_classes=num_classes).float()
    gt_displacements = torch.randn(batch_size, num_masks, seq_len, 2)
    gt_state_classes = torch.randn(batch_size, num_masks, num_state_classes)
    valid_count = [num_masks] * batch_size
    non_empty_map = torch.rand(batch_size, height, width)
    disp_valid_map = torch.rand(batch_size, seq_len, height, width, 1)
    # 初始化损失计算类
    criterion = InstanceSegmentationLoss(matcher=dummy_matcher)

    # 计算损失 TODO: add state
    losses = criterion(pred_masks, pred_classes, pred_displacements, pred_states,
                       gt_instance_masks, gt_classes, gt_displacements, gt_state_classes,
                       valid_count, non_empty_map=non_empty_map, disp_valid_map=disp_valid_map)

    print("Losses:", losses)
