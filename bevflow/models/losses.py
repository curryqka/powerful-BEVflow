"""MotionNet 损失。从上游 train_multi_seq.py 逐行搬运。

行为等价性说明：
- 上游用模块级全局变量控制开关（use_bg_temporal_consistency 等），这里改为 LossConfig 传参。
- 上游 compute_and_bp_loss 内部做 zero_grad/backward/step；这里拆成纯函数 compute_losses，
  backward 与 step 由 trainer 负责，数值路径不变。
- 与上游一致，网格尺寸 256 与类别数 5 为硬编码；权重向量 [0.005, 1, 1, 1, 1] 见 _BG_WEIGHT。
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_BG_WEIGHT = 0.005  # 背景格子损失权重，与上游 weight_vector[0] 一致
_HEIGHT_FEAT = 13
_CELL_CATEGORY_NUM = 5


@dataclass
class LossConfig:
    use_bg_tc: bool = False
    use_fg_tc: bool = False
    use_sc: bool = False
    nn_sampling: bool = False
    reg_weight_bg_tc: float = 0.1
    reg_weight_fg_tc: float = 2.5
    reg_weight_sc: float = 15.0
    # 以下为上游常量，不在配置里暴露
    use_weighted_loss: bool = True
    pred_adj_frame_distance: bool = True
    trans_matrix_idx: int = 1
    future_frames_num: int = 20


def compute_losses(
    cfg: LossConfig,
    disp_gt,          # (batch, adj, F, h, w, 2)
    valid_pixel_maps,  # (batch, adj, F, h, w)
    pixel_cat_map_gt,  # (batch, adj, h, w, cat)
    disp_pred,        # (batch*adj, F, h, w, 2)
    non_empty_map,    # (batch, adj, h, w)
    class_pred,       # (batch*adj, cat, h, w)
    motion_gt,        # (batch, adj, h, w, 2)
    motion_pred,      # (batch*adj, 2, h, w)
    trans_matrices,   # (batch, adj, F, 4, 4)
    pixel_instance_map,  # (batch, adj, h, w) uint8
    device,
):
    """返回 (total_loss 张量, 各分项 float 列表 [disp, class, motion, bg_tc, sc, fg_tc])。"""
    future_frames_num = cfg.future_frames_num

    all_disp_field_gt = disp_gt.view(-1, future_frames_num, 256, 256, 2)
    gt = all_disp_field_gt[:, -future_frames_num:, ...].contiguous()
    gt = gt.view(-1, gt.size(2), gt.size(3), gt.size(4))
    gt = gt.permute(0, 3, 1, 2).to(device)

    all_valid = valid_pixel_maps.view(-1, future_frames_num, 256, 256)
    valid_pixel_maps = all_valid[:, -future_frames_num:, ...].contiguous()
    valid_pixel_maps = valid_pixel_maps.view(-1, valid_pixel_maps.size(2), valid_pixel_maps.size(3))
    valid_pixel_maps = torch.unsqueeze(valid_pixel_maps, 1)
    valid_pixel_maps = valid_pixel_maps.to(device)

    valid_pixel_num = torch.nonzero(valid_pixel_maps).size(0)
    if valid_pixel_num == 0:
        return None, [None] * 6

    if cfg.pred_adj_frame_distance:
        disp_pred = disp_pred.view(-1, future_frames_num, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))

        bg_tc_loss = None
        instance_spatio_temp_loss = None
        instance_spatial_loss_value = None
        instance_temporal_loss_value = None
        if cfg.use_bg_tc:
            bg_tc_loss = background_temporal_consistency_loss(cfg, disp_pred, pixel_cat_map_gt, non_empty_map, trans_matrices)
        if cfg.use_fg_tc or cfg.use_sc:
            instance_spatio_temp_loss, instance_spatial_loss_value, instance_temporal_loss_value = (
                instance_spatial_temporal_consistency_loss(cfg, disp_pred, pixel_instance_map)
            )

        for c in range(1, disp_pred.size(1)):
            disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]
        disp_pred = disp_pred.view(-1, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))

    pixel_cat_map_gt = pixel_cat_map_gt.view(-1, 256, 256, _CELL_CATEGORY_NUM)

    criterion = nn.SmoothL1Loss(reduction="none")
    if cfg.use_weighted_loss:
        pixel_cat_map_gt_numpy = pixel_cat_map_gt.numpy()
        pixel_cat_map_gt_numpy = np.argmax(pixel_cat_map_gt_numpy, axis=-1) + 1
        cat_weight_map = np.zeros_like(pixel_cat_map_gt_numpy, dtype=np.float32)
        weight_vector = [_BG_WEIGHT, 1.0, 1.0, 1.0, 1.0]  # [bg, car & bus, ped, bike, other]
        for k in range(len(weight_vector)):
            mask = pixel_cat_map_gt_numpy == (k + 1)
            cat_weight_map[mask] = weight_vector[k]

        cat_weight_map = cat_weight_map[:, np.newaxis, np.newaxis, ...]  # (batch, 1, 1, h, w)
        cat_weight_map = torch.from_numpy(cat_weight_map).to(device)
        map_shape = cat_weight_map.size()

        loss_disp = criterion(gt * valid_pixel_maps, disp_pred * valid_pixel_maps)
        loss_disp = loss_disp.reshape(map_shape[0], -1, map_shape[-3], map_shape[-2], map_shape[-1])
        loss_disp = torch.sum(loss_disp * cat_weight_map) / valid_pixel_num
    else:
        loss_disp = criterion(gt * valid_pixel_maps, disp_pred * valid_pixel_maps) / valid_pixel_num

    # -- Compute the grid cell classification loss
    non_empty_map = non_empty_map.view(-1, 256, 256)
    non_empty_map = non_empty_map.to(device)
    pixel_cat_map_gt = pixel_cat_map_gt.permute(0, 3, 1, 2).to(device)

    log_softmax_probs = F.log_softmax(class_pred, dim=1)

    if cfg.use_weighted_loss:
        map_shape = cat_weight_map.size()
        cat_weight_map = cat_weight_map.view(map_shape[0], map_shape[-2], map_shape[-1])  # (bs, h, w)
        loss_class = torch.sum(-pixel_cat_map_gt * log_softmax_probs, dim=1) * cat_weight_map
    else:
        loss_class = torch.sum(-pixel_cat_map_gt * log_softmax_probs, dim=1)
    loss_class = torch.sum(loss_class * non_empty_map) / torch.nonzero(non_empty_map).size(0)

    # -- Compute the speed level classification loss
    motion_gt = motion_gt.view(-1, 256, 256, 2)
    motion_gt_numpy = motion_gt.numpy()
    motion_gt = motion_gt.permute(0, 3, 1, 2).to(device)
    log_softmax_motion_pred = F.log_softmax(motion_pred, dim=1)

    if cfg.use_weighted_loss:
        motion_gt_numpy = np.argmax(motion_gt_numpy, axis=-1) + 1
        motion_weight_map = np.zeros_like(motion_gt_numpy, dtype=np.float32)
        weight_vector = [_BG_WEIGHT, 1.0]  # [static, moving]
        for k in range(len(weight_vector)):
            mask = motion_gt_numpy == (k + 1)
            motion_weight_map[mask] = weight_vector[k]

        motion_weight_map = torch.from_numpy(motion_weight_map).to(device)
        loss_speed = torch.sum(-motion_gt * log_softmax_motion_pred, dim=1) * motion_weight_map
    else:
        loss_speed = torch.sum(-motion_gt * log_softmax_motion_pred, dim=1)
    loss_motion = torch.sum(loss_speed * non_empty_map) / torch.nonzero(non_empty_map).size(0)

    # -- Sum up all the losses
    if cfg.use_bg_tc and (cfg.use_fg_tc or cfg.use_sc):
        total = loss_disp + loss_class + loss_motion + cfg.reg_weight_bg_tc * bg_tc_loss + instance_spatio_temp_loss
    elif cfg.use_bg_tc:
        total = loss_disp + loss_class + loss_motion + cfg.reg_weight_bg_tc * bg_tc_loss
    elif cfg.use_sc or cfg.use_fg_tc:
        total = loss_disp + loss_class + loss_motion + instance_spatio_temp_loss
    else:
        total = loss_disp + loss_class + loss_motion

    bg_tc_value = bg_tc_loss.item() if cfg.use_bg_tc else -1
    if cfg.use_sc or cfg.use_fg_tc:
        sc_value = instance_spatial_loss_value
        fg_tc_value = instance_temporal_loss_value
    else:
        sc_value = -1
        fg_tc_value = -1

    return total, [loss_disp.item(), loss_class.item(), loss_motion.item(), bg_tc_value, sc_value, fg_tc_value]


def background_temporal_consistency_loss(cfg: LossConfig, disp_pred, pixel_cat_map_gt, non_empty_map, trans_matrices):
    """
    disp_pred: relative displacement between adjacent frames. shape (batch * 2, sweep_num, 2, h, w)
    pixel_cat_map_gt: Shape (batch, 2, h, w, cat_num)
    non_empty_map: Shape (batch, 2, h, w)
    trans_matrices: Shape (batch, 2, sweep_num, 4, 4)
    """
    criterion = nn.SmoothL1Loss(reduction="sum")

    non_empty_map_numpy = non_empty_map.numpy()
    pixel_cat_maps = pixel_cat_map_gt.numpy()
    max_prob = np.amax(pixel_cat_maps, axis=-1)
    filter_mask = max_prob == 1.0
    pixel_cat_maps = np.argmax(pixel_cat_maps, axis=-1) + 1  # category starts from 1 (background), etc
    pixel_cat_maps = pixel_cat_maps * non_empty_map_numpy * filter_mask  # (batch, 2, h, w)

    trans_matrices = trans_matrices.numpy()
    device = disp_pred.device

    pred_shape = disp_pred.size()
    disp_pred = disp_pred.view(-1, 2, pred_shape[1], pred_shape[2], pred_shape[3], pred_shape[4])

    seq_1_pred = disp_pred[:, 0]  # (batch, sweep_num, 2, h, w)
    seq_2_pred = disp_pred[:, 1]

    seq_1_absolute_pred_list = list()
    seq_2_absolute_pred_list = list()

    seq_1_absolute_pred_list.append(seq_1_pred[:, 1])
    for i in range(2, pred_shape[1]):
        seq_1_absolute_pred_list.append(seq_1_pred[:, i] + seq_1_absolute_pred_list[i - 2])

    seq_2_absolute_pred_list.append(seq_2_pred[:, 0])
    for i in range(1, pred_shape[1] - 1):
        seq_2_absolute_pred_list.append(seq_2_pred[:, i] + seq_2_absolute_pred_list[i - 1])

    # -- Compute the transformation matrices
    transformed_disp_pred_list = list()

    trans_matrix_global = trans_matrices[:, 1]  # (batch, sweep_num, 4, 4)
    trans_matrix_global = trans_matrix_global[:, cfg.trans_matrix_idx, 0:3]  # (batch, 3, 4)
    trans_matrix_global = trans_matrix_global[:, :, (0, 1, 3)]  # (batch, 3, 3)
    trans_matrix_global[:, 2] = np.array([0.0, 0.0, 1.0])

    translate_to_global = np.array([[1.0, 0.0, -120.0], [0.0, 1.0, -120.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    scale_global = np.array([[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    trans_global = scale_global @ translate_to_global
    inv_trans_global = np.linalg.inv(trans_global)

    trans_global = np.expand_dims(trans_global, axis=0)
    inv_trans_global = np.expand_dims(inv_trans_global, axis=0)
    trans_matrix_total = inv_trans_global @ trans_matrix_global @ trans_global

    w, h = pred_shape[-2], pred_shape[-1]
    resize_m = np.array([[2 / w, 0.0, -1], [0.0, 2 / h, -1], [0.0, 0.0, 1]], dtype=np.float32)
    inverse_m = np.linalg.inv(resize_m)
    resize_m = np.expand_dims(resize_m, axis=0)
    inverse_m = np.expand_dims(inverse_m, axis=0)

    grid_trans_matrix = resize_m @ trans_matrix_total @ inverse_m  # (batch, 3, 3)
    grid_trans_matrix = grid_trans_matrix[:, :2].astype(np.float32)
    grid_trans_matrix = torch.from_numpy(grid_trans_matrix)

    trans_matrix_translation_global = np.eye(trans_matrix_total.shape[1])
    trans_matrix_translation_global = np.expand_dims(trans_matrix_translation_global, axis=0)
    trans_matrix_translation_global = np.repeat(trans_matrix_translation_global, grid_trans_matrix.shape[0], axis=0)
    trans_matrix_translation_global[:, :, 2] = trans_matrix_global[:, :, 2]  # only translation
    trans_matrix_translation_total = inv_trans_global @ trans_matrix_translation_global @ trans_global

    grid_trans_matrix_disp = resize_m @ trans_matrix_translation_total @ inverse_m
    grid_trans_matrix_disp = grid_trans_matrix_disp[:, :2].astype(np.float32)
    grid_trans_matrix_disp = torch.from_numpy(grid_trans_matrix_disp).to(device)

    disp_rotate_matrix = trans_matrix_global[:, 0:2, 0:2].astype(np.float32)  # (batch, 2, 2)
    disp_rotate_matrix = torch.from_numpy(disp_rotate_matrix).to(device)

    for i in range(len(seq_1_absolute_pred_list)):
        curr_pred = seq_1_absolute_pred_list[i]  # (batch, 2, h, w)

        # rotation
        curr_pred = curr_pred.permute(0, 2, 3, 1).contiguous()  # (batch, h, w, 2)
        curr_pred = curr_pred.view(-1, h * w, 2)
        curr_pred = torch.bmm(curr_pred, disp_rotate_matrix)
        curr_pred = curr_pred.view(-1, h, w, 2)
        curr_pred = curr_pred.permute(0, 3, 1, 2).contiguous()  # (batch, 2, h, w)

        # translation
        curr_pred = curr_pred.permute(0, 1, 3, 2).contiguous()  # swap x and y axis
        curr_pred = torch.flip(curr_pred, dims=[2])

        grid = F.affine_grid(grid_trans_matrix_disp, curr_pred.size())
        if cfg.nn_sampling:
            curr_pred = F.grid_sample(curr_pred, grid, mode="nearest")
        else:
            curr_pred = F.grid_sample(curr_pred, grid)

        curr_pred = torch.flip(curr_pred, dims=[2])
        curr_pred = curr_pred.permute(0, 1, 3, 2).contiguous()

        transformed_disp_pred_list.append(curr_pred)

    # -- Transform the category map
    pixel_cat_map = pixel_cat_maps[:, 0]  # (batch, h, w)
    pixel_cat_map = torch.from_numpy(pixel_cat_map.astype(np.float32))
    pixel_cat_map = pixel_cat_map[:, None, :, :]  # (batch, 1, h, w)
    trans_pixel_cat_map = pixel_cat_map.permute(0, 1, 3, 2)  # swap x and y axis
    trans_pixel_cat_map = torch.flip(trans_pixel_cat_map, dims=[2])

    grid = F.affine_grid(grid_trans_matrix, pixel_cat_map.size())
    trans_pixel_cat_map = F.grid_sample(trans_pixel_cat_map, grid, mode="nearest")

    trans_pixel_cat_map = torch.flip(trans_pixel_cat_map, dims=[2])
    trans_pixel_cat_map = trans_pixel_cat_map.permute(0, 1, 3, 2)

    adj_pixel_cat_map = pixel_cat_maps[:, 1]
    adj_pixel_cat_map = torch.from_numpy(adj_pixel_cat_map.astype(np.float32))
    adj_pixel_cat_map = torch.unsqueeze(adj_pixel_cat_map, dim=1)

    mask_common = trans_pixel_cat_map == adj_pixel_cat_map
    mask_common = mask_common.float()
    non_empty_map_gpu = non_empty_map.to(device)
    non_empty_map_gpu = non_empty_map_gpu[:, 1:2, :, :]  # select the second sequence, keep dim
    mask_common = mask_common.to(device)
    mask_common = mask_common * non_empty_map_gpu

    loss_list = list()
    for i in range(len(seq_1_absolute_pred_list)):
        trans_seq_1_pred = transformed_disp_pred_list[i]  # (batch, 2, h, w)
        seq_2_pred = seq_2_absolute_pred_list[i]  # (batch, 2, h, w)

        trans_seq_1_pred = trans_seq_1_pred * mask_common
        seq_2_pred = seq_2_pred * mask_common

        num_non_empty_cells = torch.nonzero(mask_common).size(0)
        if num_non_empty_cells != 0:
            loss = criterion(trans_seq_1_pred, seq_2_pred) / num_non_empty_cells
            loss_list.append(loss)

    res_loss = torch.mean(torch.stack(loss_list, 0))
    return res_loss


def instance_spatial_temporal_consistency_loss(cfg: LossConfig, disp_pred, pixel_instance_map):
    device = disp_pred.device

    pred_shape = disp_pred.size()
    disp_pred = disp_pred.view(-1, 2, pred_shape[1], pred_shape[2], pred_shape[3], pred_shape[4])

    seq_1_pred = disp_pred[:, 0]  # (batch, sweep_num, 2, h, w)
    seq_2_pred = disp_pred[:, 1]

    pixel_instance_map = pixel_instance_map.numpy()
    batch = pixel_instance_map.shape[0]

    spatial_loss = 0.0
    temporal_loss = 0.0
    counter = 0
    criterion = nn.SmoothL1Loss()

    for i in range(batch):
        curr_batch_instance_maps = pixel_instance_map[i]
        seq_1_instance_map = curr_batch_instance_maps[0]
        seq_2_instance_map = curr_batch_instance_maps[1]

        seq_1_instance_ids = np.unique(seq_1_instance_map)
        seq_2_instance_ids = np.unique(seq_2_instance_map)
        common_instance_ids = np.intersect1d(seq_1_instance_ids, seq_2_instance_ids, assume_unique=True)

        seq_1_batch_pred = seq_1_pred[i]  # (sweep_num, 2, h, w)
        seq_2_batch_pred = seq_2_pred[i]

        for h in common_instance_ids:
            if h == 0:  # do not consider the background instance
                continue

            seq_1_mask = np.where(seq_1_instance_map == h)
            seq_1_idx_x = torch.from_numpy(seq_1_mask[0]).to(device)
            seq_1_idx_y = torch.from_numpy(seq_1_mask[1]).to(device)
            seq_1_selected_cells = seq_1_batch_pred[:, :, seq_1_idx_x, seq_1_idx_y]

            seq_2_mask = np.where(seq_2_instance_map == h)
            seq_2_idx_x = torch.from_numpy(seq_2_mask[0]).to(device)
            seq_2_idx_y = torch.from_numpy(seq_2_mask[1]).to(device)
            seq_2_selected_cells = seq_2_batch_pred[:, :, seq_2_idx_x, seq_2_idx_y]

            seq_1_selected_cell_num = seq_1_selected_cells.size(2)
            seq_2_selected_cell_num = seq_2_selected_cells.size(2)

            if cfg.use_sc:
                tmp_seq_1 = 0
                if seq_1_selected_cell_num > 1:
                    tmp_seq_1 = criterion(seq_1_selected_cells[:, :, :-1], seq_1_selected_cells[:, :, 1:])

                tmp_seq_2 = 0
                if seq_2_selected_cell_num > 1:
                    tmp_seq_2 = criterion(seq_2_selected_cells[:, :, :-1], seq_2_selected_cells[:, :, 1:])

                spatial_loss += tmp_seq_1 + tmp_seq_2

            if cfg.use_fg_tc:
                seq_1_mean = torch.mean(seq_1_selected_cells, dim=2)
                seq_2_mean = torch.mean(seq_2_selected_cells, dim=2)
                temporal_loss += criterion(seq_1_mean, seq_2_mean)

            counter += 1

    if counter != 0:
        spatial_loss = spatial_loss / counter
        temporal_loss = temporal_loss / counter

    total_loss = cfg.reg_weight_sc * spatial_loss + cfg.reg_weight_fg_tc * temporal_loss

    spatial_loss_value = 0 if type(spatial_loss) is float else spatial_loss.item()
    temporal_loss_value = 0 if type(temporal_loss) is float else temporal_loss.item()

    return total_loss, spatial_loss_value, temporal_loss_value
