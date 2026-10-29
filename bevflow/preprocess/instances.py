# SPDX-License-Identifier: Apache-2.0
"""VpE 的实例序列提取。从旧代码 data/instance_extractor.py 的 get_instance_infoV2 搬运：
按实例取像素坐标，凸包内插值 num_grid 个点与其末帧位移，类别 one-hot。

输入（单批，numpy）：
- all_disp_field_gt: (B, T, 2, H, W)，末帧位移用于插值
- pixel_cat_maps_gt:  (B, C, H, W)
- pixel_instance_maps_gt: (B, H, W)
输出：三个 list，每元素 (N_ins, num_grid, 2) / (N_ins, num_grid, 2) / (N_ins, C)。
"""
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError


def get_instance_info(all_disp_field_gt, pixel_cat_maps_gt, pixel_instance_maps_gt, num_grid=20):
    last_disp = all_disp_field_gt[:, -1, ...]  # (B, 2, H, W)
    batch_positions, batch_displacements, batch_categories = [], [], []
    batch_size, h, w = pixel_instance_maps_gt.shape

    for b in range(batch_size):
        positions_list, disp_list, cat_list = [], [], []
        instance_map = pixel_instance_maps_gt[b]
        unique_instances = np.unique(instance_map)
        unique_instances = unique_instances[unique_instances != 0]

        for instance_id in unique_instances:
            positions = np.argwhere(instance_map == instance_id)
            if len(np.unique(positions[:, 0])) < 2 or len(np.unique(positions[:, 1])) < 2:
                interp_positions = np.tile(positions[0], (num_grid, 1))
                interp_disp = np.tile(last_disp[b, :, positions[0][0], positions[0][1]], (num_grid, 1))
            else:
                try:
                    hull = ConvexHull(positions)
                    hull_points = positions[hull.vertices]
                    x_min, y_min = hull_points[:, 1].min(), hull_points[:, 0].min()
                    x_max, y_max = hull_points[:, 1].max(), hull_points[:, 0].max()
                    grid_x, grid_y = np.meshgrid(
                        np.linspace(x_min, x_max, num_grid), np.linspace(y_min, y_max, num_grid))
                    interp_positions = griddata(
                        points=hull_points, values=hull_points, xi=(grid_y, grid_x),
                        method="linear", fill_value=0)
                    disp_x = griddata(
                        points=hull_points,
                        values=last_disp[b, 0, hull_points[:, 0], hull_points[:, 1]],
                        xi=(grid_y, grid_x), method="linear", fill_value=0)
                    disp_y = griddata(
                        points=hull_points,
                        values=last_disp[b, 1, hull_points[:, 0], hull_points[:, 1]],
                        xi=(grid_y, grid_x), method="linear", fill_value=0)
                    interp_disp = np.stack((disp_x, disp_y), axis=-1)
                except QhullError:
                    interp_positions = np.tile(positions[0], (num_grid, 1))
                    interp_disp = np.tile(last_disp[b, :, positions[0][0], positions[0][1]],
                                          (num_grid, 1))
                interp_positions = np.nan_to_num(interp_positions, nan=0.0)
                interp_disp = np.nan_to_num(interp_disp, nan=0.0)

            interp_positions = _fit_len(interp_positions, num_grid)
            interp_disp = _fit_len(interp_disp, num_grid)
            positions_list.append(interp_positions)
            disp_list.append(interp_disp)

            category = pixel_cat_maps_gt[b, :, positions[0][0], positions[0][1]]
            cat_list.append((category == np.arange(pixel_cat_maps_gt.shape[1])).astype(np.float32))

        batch_positions.append(
            np.stack(positions_list, axis=0) if positions_list
            else np.zeros([0, num_grid, 2], dtype=np.float32))
        batch_displacements.append(
            np.stack(disp_list, axis=0) if disp_list
            else np.zeros([0, num_grid, 2], dtype=np.float32))
        batch_categories.append(
            np.stack(cat_list, axis=0) if cat_list
            else np.zeros([0, pixel_cat_maps_gt.shape[1]], dtype=np.float32))

    return batch_positions, batch_displacements, batch_categories


def _fit_len(arr, num_grid):
    arr = arr.reshape(-1, 2)
    if arr.shape[0] < num_grid:
        return np.pad(arr, ((0, num_grid - arr.shape[0]), (0, 0)), mode="constant")
    return arr[:num_grid]


def get_instance_info_paper(disp_gt, cat_gt, inst_map, num_grid=20, voxel_xy=0.25):
    """论文式5 口径的实例序列：沿实例轨迹采样 num_grid 个点 + 逐帧位移序列。

    disp_gt: (B, T, 2, H, W) 米；cat_gt: (B, C, H, W)；inst_map: (B, H, W)。
    返回每样本 (traj (N_ins,num_grid,2) 像素坐标, disp_seq (N_ins,T,2) 米, onehot (N_ins,C))。
    轨迹 = 实例质心 + 逐帧平均位移的累积（米→像素 /0.25）。
    """
    B, T, _, H, W = disp_gt.shape
    out_traj, out_disp, out_cat = [], [], []
    for b in range(B):
        traj_list, disp_list, cat_list = [], [], []
        ids = np.unique(inst_map[b])
        ids = ids[ids != 0]
        for iid in ids:
            cells = np.argwhere(inst_map[b] == iid)          # (M,2)
            # 高级索引结果 (M,T,2)，对格子取均值 → (T,2) 米
            seq = disp_gt[b, :, :, cells[:, 0], cells[:, 1]].mean(axis=0)
            centroid = cells.mean(axis=0)                    # 像素 (row,col)
            path = centroid[None, :] + np.cumsum(seq, axis=0) / voxel_xy     # (T,2)
            # 沿折线等弧长重采样 num_grid 点
            seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            target = np.linspace(0, s[-1], num_grid)
            pts = np.stack([np.interp(target, s, path[:, d]) for d in range(2)], axis=-1)
            traj_list.append(pts.astype(np.float32))
            disp_list.append(seq.astype(np.float32))
            cat_list.append((cat_gt[b, :, cells[0, 0], cells[0, 1]] == np.arange(cat_gt.shape[1])).astype(np.float32))
        out_traj.append(np.stack(traj_list) if traj_list else np.zeros([0, num_grid, 2], np.float32))
        out_disp.append(np.stack(disp_list) if disp_list else np.zeros([0, T, 2], np.float32))
        out_cat.append(np.stack(cat_list) if cat_list else np.zeros([0, cat_gt.shape[1]], np.float32))
    return out_traj, out_disp, out_cat
