# -*- coding: utf-8 -*-

# 这样也行
# coding:utf-8 

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
import scipy
def reshape_disp(disp_pred, future_frames_num=20):
    """
    Reshape the displacement prediction to be used for warping.
    """
    if disp_pred.shape[1] == future_frames_num * 2:
        disp_pred = disp_pred.view(disp_pred.shape[0], future_frames_num, 2, disp_pred.shape[2], disp_pred.shape[3]).contiguous()
    # disp_pred = disp_pred.view(-1, future_frames_num, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))
   
    for c in range(1, disp_pred.size(1)):
        disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]
    # disp_pred = disp_pred.view(disp_pred.shape[0], -1, disp_pred.shape[2], disp_pred.shape[3])
    disp_pred =disp_pred[:, -1, ...]
    return disp_pred

def reshape_dispV2(disp_pred, valid_maps, future_frames_num=20):
    """
    Reshape the displacement prediction to be used for warping.
    Args:
        -disp_pred: [bs, seq, 2, h, w]
        -valid_map: [bs, seq, 1, h, w]
    """
    if disp_pred.shape[1] == future_frames_num * 2:
        disp_pred = disp_pred.view(disp_pred.shape[0], future_frames_num, 2, disp_pred.shape[2], disp_pred.shape[3]).contiguous()
    
    for c in range(1, disp_pred.size(1)):
        disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]
    # breakpoint()
    disp_pred = disp_pred * valid_maps
    disp_pred_clone = disp_pred
    disp_pred_clone = disp_pred_clone.view(disp_pred.size(0), -1, disp_pred.size(-2), disp_pred.size(-1))
    return disp_pred_clone

def reshape_dispV3(disp, valid_maps, future_frames_num=20, gt_flag=False):
    """
    Reshape the displacement labels to be used for warping.
    """
    
    # 确保 disp 是 NumPy 数组
    if not isinstance(disp, np.ndarray):
        raise ValueError("Input 'disp' must be a NumPy array.")
    
    # 重塑 disp 的形状
    if disp.shape[1] == future_frames_num * 2:
        disp = disp.reshape(disp.shape[0], future_frames_num, 2, disp.shape[2], disp.shape[3])
    
    disp_warp = disp.copy()
    
    if gt_flag:
        for c in range(1, disp.shape[1]):
            disp_warp[:, c, ...] = disp[:, c, ...] - disp[:, c - 1, ...]
        disp_warp = disp_warp * valid_maps
    else:
        disp_warp = disp_warp * valid_maps
    
    # 重塑 disp_warp 的形状
    disp_warp = disp_warp.reshape(disp_warp.shape[0], disp.shape[1], -1, disp_warp.shape[-2], disp_warp.shape[-1])
    
    return disp_warp

# def get_instance_info(all_disp_field_gt, valid_maps, pixel_cat_maps_gt, pixel_instance_maps_gt):
    
#     last_disp_field_gt = all_disp_field_gt[:, -1, ...]
#     import numpy as np


def get_instance_info(all_disp_field_gt, valid_maps, pixel_cat_maps_gt, pixel_instance_maps_gt, num_grid=20):
    """
    Args: all_disp_field_gt [bs, seq, 2, h, w]
          valid_maps [bs, seq, 1, h, w]
          pixel_cat_maps_gt [bs, cell_cls, h, w]
          pixel_instance_maps_gt [bs, h, w]
    """
    # 获取最后一帧的位移场
    last_disp_field_gt = all_disp_field_gt[:, -1, ...]
    
    # 存储结果的列表
    # instance_positions = []
    # instance_categories = []
    # instance_displacements = []

    batch_instance_positions = []
    batch_instance_displacements = []
    batch_instance_categories = []
    batch_size, h, w = pixel_instance_maps_gt.shape
    
    for b in range(batch_size):
        instance_positions = []
        instance_displacements = []
        instance_categories = []
        # 获取当前批次的实例标记
        instance_map = pixel_instance_maps_gt[b]
        
        # 获取唯一的实例ID（排除背景0）
        unique_instances = np.unique(instance_map)
        unique_instances = unique_instances[unique_instances != 0]
        
        for instance_id in unique_instances:
            # 获取该实例的坐标
            positions = np.argwhere(instance_map == instance_id)
            
            # 插值到固定数量的网格点
            if len(positions) > 1:
                interp_func_x = interp1d(np.arange(len(positions)), positions[:, 0], kind='linear', fill_value="extrapolate")
                interp_func_y = interp1d(np.arange(len(positions)), positions[:, 1], kind='linear', fill_value="extrapolate")
                new_indices = np.linspace(0, len(positions) - 1, num_grid)
                interpolated_positions = np.stack((interp_func_x(new_indices), interp_func_y(new_indices)), axis=-1)
            else:
                interpolated_positions = np.tile(positions, (num_grid, 1))
            
            instance_positions.append(interpolated_positions)

            
            # 获取该实例的类别并进行one-hot编码
            category = pixel_cat_maps_gt[b, :, positions[0][0], positions[0][1]]
            # breakpoint()
            one_hot_category = (category == np.arange(pixel_cat_maps_gt.shape[1])).astype(np.float32)
            instance_categories.append(one_hot_category)
            
            # 获取该实例的位移
            displacements = last_disp_field_gt[b, :, positions[:, 0], positions[:, 1]]
            if len(displacements) > 1:
                interp_func_disp_x = interp1d(np.arange(len(positions)), displacements[:, 0], kind='linear', fill_value="extrapolate")
                interp_func_disp_y = interp1d(np.arange(len(positions)), displacements[:, 1], kind='linear', fill_value="extrapolate")
                interpolated_displacements = np.stack((interp_func_disp_x(new_indices), interp_func_disp_y(new_indices)), axis=-1)
            else:
                interpolated_displacements = np.tile(displacements, (num_grid, 1))
            
            instance_displacements.append(interpolated_displacements)
        # breakpoint()
        # then stack: num_polylines x n_points x 2
        # sd_map_graph = np.stack(sd_map_graph, axis=0) if len(sd_map_graph) > 0 \
        #     else np.zeros([0, n_points, 2], dtype=np.float32)
        # onehot_category = np.concatenate(onehot_category, axis=0) if len(onehot_category) > 0 \
        #     else np.zeros([0, num_categories], dtype=np.float32)
        single_sample_instance_positions = np.stack(instance_positions, axis=0) if len(instance_positions) > 0 \
            else np.zeros([0, num_grid, 2], dtype=np.float32)
        single_sample_instance_displacements = np.stack(instance_displacements, axis=0) if len(instance_displacements) > 0 \
            else np.zeros([0, num_grid, 2], dtype=np.float32)
        single_sample_instance_categories = np.stack(instance_categories, axis=0) if len(instance_categories) > 0 \
            else np.zeros([0, pixel_cat_maps_gt.shape[1]], dtype=np.float32)
        batch_instance_positions.append(single_sample_instance_positions)
        batch_instance_displacements.append(single_sample_instance_displacements)
        batch_instance_categories.append(single_sample_instance_categories)
    return batch_instance_positions, batch_instance_displacements, batch_instance_categories
            

# # def get_instance_infoV2(all_disp_field_gt, valid_maps, pixel_cat_maps_gt, pixel_instance_maps_gt, num_grid=20):
#     # 获取最后一帧的位移场
#     last_disp_field_gt = all_disp_field_gt[:, -1, ...]

#     # 存储结果的列表
#     batch_instance_positions = []
#     batch_instance_displacements = []
#     batch_instance_categories = []
    
#     # 获取batch的大小，以及图像的高和宽
#     batch_size, h, w = pixel_instance_maps_gt.shape
    
#     for b in range(batch_size):
#         instance_positions = []
#         instance_displacements = []
#         instance_categories = []
        
#         # 获取当前批次的实例标记
#         instance_map = pixel_instance_maps_gt[b]
        
#         # 获取唯一的实例ID（排除背景0）
#         unique_instances = np.unique(instance_map)
#         unique_instances = unique_instances[unique_instances != 0]
        
#         for instance_id in unique_instances:
#             # 获取该实例的坐标
#             positions = np.argwhere(instance_map == instance_id)
            
#             # 检查数据是否适合二维插值
#             if len(positions) < 4 or len(np.unique(positions[:, 0])) < 2 or len(np.unique(positions[:, 1])) < 2:
#                 # 数据不适合二维插值，使用最近邻插值或其他方法
#                 interpolated_positions = np.tile(positions[0], (num_grid, 1))
#             else:
#                 # 生成一个规则的网格，用于插值
#                 grid_x, grid_y = np.meshgrid(
#                     np.linspace(positions[:, 1].min(), positions[:, 1].max(), num_grid),
#                     np.linspace(positions[:, 0].min(), positions[:, 0].max(), num_grid)
#                 )

#                 # 使用 griddata 进行二维插值
#                 try:
#                     interpolated_positions = griddata(
#                         points=positions, 
#                         values=positions,  # positions 本身作为值
#                         xi=(grid_y, grid_x), 
#                         method='linear',
#                         fill_value=0  # 设置填充值，防止出现 NaN
#                     )
#                 except scipy.spatial._qhull.QhullError:
#                     # 插值失败，使用最近邻插值
#                     interpolated_positions = np.tile(positions[0], (num_grid, 1))
                
#                 # 处理插值结果中的 NaN 值
#                 interpolated_positions = np.nan_to_num(interpolated_positions, nan=0.0)
            
#             # 确保插值后的点数一致，填充或裁剪到 num_grid 大小
#             interpolated_positions = interpolated_positions.reshape(-1, 2)
#             if interpolated_positions.shape[0] < num_grid:
#                 padding = num_grid - interpolated_positions.shape[0]
#                 interpolated_positions = np.pad(interpolated_positions, ((0, padding), (0, 0)), mode='constant')
#             elif interpolated_positions.shape[0] > num_grid:
#                 interpolated_positions = interpolated_positions[:num_grid]
            
#             instance_positions.append(interpolated_positions)

#             # 获取该实例的类别并进行 one-hot 编码
#             category = pixel_cat_maps_gt[b, :, positions[0][0], positions[0][1]]
#             one_hot_category = (category == np.arange(pixel_cat_maps_gt.shape[1])).astype(np.float32)
#             instance_categories.append(one_hot_category)
            
#             # 获取该实例的位移
#             displacements = last_disp_field_gt[b, :, positions[:, 0], positions[:, 1]]
#             # 对位移的处理与位置相同
#             if len(positions) < 4 or len(np.unique(positions[:, 0])) < 2 or len(np.unique(positions[:, 1])) < 2:
#                 interpolated_displacements = np.tile(displacements[0], (num_grid, 1))
#             else:
#                 try:
#                     interpolated_displacements_x = griddata(
#                         points=positions, 
#                         values=displacements[:, 0],
#                         xi=(grid_y, grid_x),
#                         method='linear',
#                         fill_value=0
#                     )
#                     interpolated_displacements_y = griddata(
#                         points=positions, 
#                         values=displacements[:, 1],
#                         xi=(grid_y, grid_x),
#                         method='linear',
#                         fill_value=0
#                     )
#                     interpolated_displacements_x = np.nan_to_num(interpolated_displacements_x, nan=0.0)
#                     interpolated_displacements_y = np.nan_to_num(interpolated_displacements_y, nan=0.0)
#                     interpolated_displacements = np.stack((interpolated_displacements_x, interpolated_displacements_y), axis=-1)
#                 except scipy.spatial._qhull.QhullError:
#                     # 插值失败，使用最近邻插值
#                     interpolated_displacements = np.tile(displacements[0], (num_grid, 1))
            
#             # 确保插值后的点数一致，填充或裁剪到 num_grid 大小
#             interpolated_displacements = interpolated_displacements.reshape(-1, 2)
#             if interpolated_displacements.shape[0] < num_grid:
#                 padding = num_grid - interpolated_displacements.shape[0]
#                 interpolated_displacements = np.pad(interpolated_displacements, ((0, padding), (0, 0)), mode='constant')
#             elif interpolated_displacements.shape[0] > num_grid:
#                 interpolated_displacements = interpolated_displacements[:num_grid]
            
#             instance_displacements.append(interpolated_displacements)

#         # 将每个实例的列表转换为数组
#         single_sample_instance_positions = np.stack(instance_positions, axis=0) if len(instance_positions) > 0 \
#             else np.zeros([0, num_grid, 2], dtype=np.float32)
#         single_sample_instance_displacements = np.stack(instance_displacements, axis=0) if len(instance_displacements) > 0 \
#             else np.zeros([0, num_grid, 2], dtype=np.float32)
#         single_sample_instance_categories = np.stack(instance_categories, axis=0) if len(instance_categories) > 0 \
#             else np.zeros([0, pixel_cat_maps_gt.shape[1]], dtype=np.float32)
#         def check_for_nan(data):
#             if np.isnan(data).any():
#                 print("Data contains NaN values.")
#             else:
#                 # print("Data does not contain any NaN values.")
#                 pass
#         check_for_nan(single_sample_instance_positions)
#         check_for_nan(single_sample_instance_displacements)
#         check_for_nan(single_sample_instance_categories)
#         # 记录每个batch的结果
#         batch_instance_positions.append(single_sample_instance_positions)
#         batch_instance_displacements.append(single_sample_instance_displacements)
#         batch_instance_categories.append(single_sample_instance_categories)
    
#     return batch_instance_positions, batch_instance_displacements, batch_instance_categories

def get_instance_infoV2(all_disp_field_gt, valid_maps, pixel_cat_maps_gt, pixel_instance_maps_gt, num_grid=20):
    # 获取最后一帧的位移场
    last_disp_field_gt = all_disp_field_gt[:, -1, ...]

    # 存储结果的列表
    batch_instance_positions = []
    batch_instance_displacements = []
    batch_instance_categories = []
    
    # 获取batch的大小，以及图像的高和宽
    batch_size, h, w = pixel_instance_maps_gt.shape
    
    for b in range(batch_size):
        instance_positions = []
        instance_displacements = []
        instance_categories = []
        
        # 获取当前批次的实例标记
        instance_map = pixel_instance_maps_gt[b]
        
        # 获取唯一的实例ID（排除背景0）
        unique_instances = np.unique(instance_map)
        unique_instances = unique_instances[unique_instances != 0]
        
        for instance_id in unique_instances:
            # 获取该实例的坐标
            positions = np.argwhere(instance_map == instance_id)
            
            # 检查点集是否退化
            if len(np.unique(positions[:, 0])) < 2 or len(np.unique(positions[:, 1])) < 2:
                # 退化点集，使用最近邻插值
                interpolated_positions = np.tile(positions[0], (num_grid, 1))
                interpolated_displacements = np.tile(last_disp_field_gt[b, :, positions[0][0], positions[0][1]], (num_grid, 1))
            else:
                try:
                    # 计算凸包
                    hull = ConvexHull(positions)
                    hull_points = positions[hull.vertices]
                    
                    # 生成一个规则的网格，用于插值
                    x_min, y_min = hull_points[:, 1].min(), hull_points[:, 0].min()
                    x_max, y_max = hull_points[:, 1].max(), hull_points[:, 0].max()
                    
                    grid_x, grid_y = np.meshgrid(
                        np.linspace(x_min, x_max, num_grid),
                        np.linspace(y_min, y_max, num_grid)
                    )
                    
                    # 使用 griddata 进行二维插值
                    interpolated_positions = griddata(
                        points=hull_points, 
                        values=hull_points,  # hull_points 本身作为值
                        xi=(grid_y, grid_x), 
                        method='linear',
                        fill_value=0  # 设置填充值，防止出现 NaN
                    )
                    interpolated_displacements_x = griddata(
                        points=hull_points, 
                        values=last_disp_field_gt[b, 0, hull_points[:, 0], hull_points[:, 1]],
                        xi=(grid_y, grid_x),
                        method='linear',
                        fill_value=0
                    )
                    interpolated_displacements_y = griddata(
                        points=hull_points, 
                        values=last_disp_field_gt[b, 1, hull_points[:, 0], hull_points[:, 1]],
                        xi=(grid_y, grid_x),
                        method='linear',
                        fill_value=0
                    )
                    interpolated_displacements = np.stack((interpolated_displacements_x, interpolated_displacements_y), axis=-1)
                except scipy.spatial._qhull.QhullError:
                    # 凸包计算失败，使用最近邻插值
                    interpolated_positions = np.tile(positions[0], (num_grid, 1))
                    interpolated_displacements = np.tile(last_disp_field_gt[b, :, positions[0][0], positions[0][1]], (num_grid, 1))
                
                # 处理插值结果中的 NaN 值
                interpolated_positions = np.nan_to_num(interpolated_positions, nan=0.0)
                interpolated_displacements = np.nan_to_num(interpolated_displacements, nan=0.0)
            
            # 确保插值后的点数一致，填充或裁剪到 num_grid 大小
            interpolated_positions = interpolated_positions.reshape(-1, 2)
            if interpolated_positions.shape[0] < num_grid:
                padding = num_grid - interpolated_positions.shape[0]
                interpolated_positions = np.pad(interpolated_positions, ((0, padding), (0, 0)), mode='constant')
            elif interpolated_positions.shape[0] > num_grid:
                interpolated_positions = interpolated_positions[:num_grid]
            
            interpolated_displacements = interpolated_displacements.reshape(-1, 2)
            if interpolated_displacements.shape[0] < num_grid:
                padding = num_grid - interpolated_displacements.shape[0]
                interpolated_displacements = np.pad(interpolated_displacements, ((0, padding), (0, 0)), mode='constant')
            elif interpolated_displacements.shape[0] > num_grid:
                interpolated_displacements = interpolated_displacements[:num_grid]
            
            instance_positions.append(interpolated_positions)
            instance_displacements.append(interpolated_displacements)

            # 获取该实例的类别并进行 one-hot 编码
            category = pixel_cat_maps_gt[b, :, positions[0][0], positions[0][1]]
            one_hot_category = (category == np.arange(pixel_cat_maps_gt.shape[1])).astype(np.float32)
            instance_categories.append(one_hot_category)
        
        # 将每个实例的列表转换为数组
        single_sample_instance_positions = np.stack(instance_positions, axis=0) if len(instance_positions) > 0 \
            else np.zeros([0, num_grid, 2], dtype=np.float32)
        single_sample_instance_displacements = np.stack(instance_displacements, axis=0) if len(instance_displacements) > 0 \
            else np.zeros([0, num_grid, 2], dtype=np.float32)
        single_sample_instance_categories = np.stack(instance_categories, axis=0) if len(instance_categories) > 0 \
            else np.zeros([0, pixel_cat_maps_gt.shape[1]], dtype=np.float32)
        
        # 记录每个batch的结果
        batch_instance_positions.append(single_sample_instance_positions)
        batch_instance_displacements.append(single_sample_instance_displacements)
        batch_instance_categories.append(single_sample_instance_categories)
    
    return batch_instance_positions, batch_instance_displacements, batch_instance_categories
    
def get_instance_infoV3(all_disp_field_gt, valid_maps, pixel_cat_maps_gt, pixel_instance_maps_gt, num_grid=20, future_frames_num=20):
    """
    Args: 
        all_disp_field_gt [bs, timestep, 2, h, w]: 位移场，包含多个时间步长
        valid_maps [bs, timestep, 1, h, w]: 有效性掩码
        pixel_cat_maps_gt [bs, cell_cls, h, w]: 类别标签
        pixel_instance_maps_gt [bs, h, w]: 实例标签
        num_grid: 用于插值的位置网格大小
        future_frames_num: 时间步长数量
    """
    # 对位移场进行时间步长的差分计算
    all_disp_field_diff = reshape_dispV3(all_disp_field_gt, valid_maps, future_frames_num, gt_flag=True)

    # 存储结果的列表
    batch_instance_positions = []
    batch_instance_displacements = []
    batch_instance_categories = []
    
    # 获取batch的大小，以及图像的高和宽
    batch_size, h, w = pixel_instance_maps_gt.shape
    
    for b in range(batch_size):
        instance_positions = []
        instance_displacements = []
        instance_categories = []
        
        # 获取当前批次的实例标记
        instance_map = pixel_instance_maps_gt[b]
        
        # 获取唯一的实例ID（排除背景0）
        unique_instances = np.unique(instance_map)
        unique_instances = unique_instances[unique_instances != 0]
        
        for instance_id in unique_instances:
            # 获取该实例的坐标
            positions = np.argwhere(instance_map == instance_id)
            
            # 检查点集是否退化
            if len(np.unique(positions[:, 0])) < 2 or len(np.unique(positions[:, 1])) < 2:
                # 退化点集，使用最近邻插值
                interpolated_positions = np.tile(positions[0], (num_grid, 1))
            else:
                try:
                    # 计算凸包
                    hull = ConvexHull(positions)
                    hull_points = positions[hull.vertices]
                    
                    # 生成一个规则的网格，用于插值
                    x_min, y_min = hull_points[:, 1].min(), hull_points[:, 0].min()
                    x_max, y_max = hull_points[:, 1].max(), hull_points[:, 0].max()
                    
                    grid_x, grid_y = np.meshgrid(
                        np.linspace(x_min, x_max, num_grid),
                        np.linspace(y_min, y_max, num_grid)
                    )
                    
                    # 使用 griddata 进行二维插值
                    interpolated_positions = griddata(
                        points=hull_points, 
                        values=hull_points,  # hull_points 本身作为值
                        xi=(grid_y, grid_x), 
                        method='linear',
                        fill_value=0  # 设置填充值，防止出现 NaN
                    )
                except scipy.spatial._qhull.QhullError:
                    # 凸包计算失败，使用最近邻插值
                    interpolated_positions = np.tile(positions[0], (num_grid, 1))
                
                # 处理插值结果中的 NaN 值
                interpolated_positions = np.nan_to_num(interpolated_positions, nan=0.0)
            
            # 确保插值后的点数一致，填充或裁剪到 num_grid 大小
            interpolated_positions = interpolated_positions.reshape(-1, 2)
            if interpolated_positions.shape[0] < num_grid:
                padding = num_grid - interpolated_positions.shape[0]
                interpolated_positions = np.pad(interpolated_positions, ((0, padding), (0, 0)), mode='constant')
            elif interpolated_positions.shape[0] > num_grid:
                interpolated_positions = interpolated_positions[:num_grid]
            
            instance_positions.append(interpolated_positions)

            # 计算该实例在所有时间步长上的平均位移
            instance_displacement_sum = np.zeros((future_frames_num, 2), dtype=np.float32)
            for pos in positions:
                # 计算每个位置在每个时间步长的位移
                instance_displacement_sum += all_disp_field_diff[b, :, :, pos[0], pos[1]]
            
            # 计算平均位移
            instance_displacement_avg = instance_displacement_sum / len(positions)
            instance_displacements.append(instance_displacement_avg)

            # 获取该实例的类别并进行 one-hot 编码
            category = pixel_cat_maps_gt[b, :, positions[0][0], positions[0][1]]
            one_hot_category = (category == np.arange(pixel_cat_maps_gt.shape[1])).astype(np.float32)
            instance_categories.append(one_hot_category)
        
        # 将每个实例的列表转换为数组
        single_sample_instance_positions = np.stack(instance_positions, axis=0) if len(instance_positions) > 0 \
            else np.zeros([0, num_grid, 2], dtype=np.float32)
        single_sample_instance_displacements = np.stack(instance_displacements, axis=0) if len(instance_displacements) > 0 \
            else np.zeros([0, future_frames_num, 2], dtype=np.float32)
        single_sample_instance_categories = np.stack(instance_categories, axis=0) if len(instance_categories) > 0 \
            else np.zeros([0, pixel_cat_maps_gt.shape[1]], dtype=np.float32)
        
        # 记录每个batch的结果
        batch_instance_positions.append(single_sample_instance_positions)
        batch_instance_displacements.append(single_sample_instance_displacements)
        batch_instance_categories.append(single_sample_instance_categories)
    
    return batch_instance_positions, batch_instance_displacements, batch_instance_categories
    

if __name__ == "__main__":
    # 测试数据
    all_disp_field_gt = np.random.rand(32, 3, 2, 10, 10)
    valid_maps = np.random.rand(32, 3, 1, 10, 10)
    pixel_cat_maps_gt = np.random.rand(32, 5, 10, 10)
    pixel_instance_maps_gt = np.random.randint(0, 5, size=(32, 10, 10))

    # 调用方法
    instance_positions, instance_displacements, instance_categories = get_instance_infoV3(
        all_disp_field_gt, valid_maps, pixel_cat_maps_gt, pixel_instance_maps_gt, num_grid=10
    )

    # 测试实例统计
    # 获取唯一的实例ID（排除背景0）
    unique_instances = np.unique(pixel_instance_maps_gt)
    unique_instances = unique_instances[unique_instances!= 0]

    # 统计每个实例ID的网格数量
    instance_counts = np.bincount(pixel_instance_maps_gt.flatten())
    instance_counts = instance_counts[unique_instances]
    # 断言结果的形状
    # print(instance_positions.shape)
    breakpoint()
    # assert instance_positions.shape == (2, 5, 5, 2)
    # assert instance_categories.shape == (2, 5, 5)
    # assert instance_displacements.shape == (2, 5, 5, 2)