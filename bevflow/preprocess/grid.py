# SPDX-License-Identifier: AGPL-3.0-or-later
# 本文件派生自 MERL MotionNet（https://github.com/merlresearch/MotionNet）。
"""BEV 网格与 GT 生成。从上游 data/data_utils.py 原样搬运本项目路径用到的函数。

未搬运：voxelize（含 numba padding_voxel）、voxelize_pillar_indices、voxelize_point_pillar、
padding_point_pillar、compute_ratio_cat_and_motion、compute_speed_level、get_static_and_moving_cells。
它们属于体素/支柱变体与调参工具，MotionNet 数据生产路径不经过，见 docs/design/upstream_map.md。
"""
import numpy as np
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


def point_in_hull_fast(points: np.array, bounding_box: Box):
    """旋转点云与包围盒到轴对齐后做成员判断。返回盒内点索引。"""
    bounding_box.orientation = bounding_box.orientation.normalised

    pc = bounding_box.orientation.inverse.rotation_matrix @ points.T
    pc = pc.T

    orientation_backup = Quaternion(bounding_box.orientation)
    bounding_box.rotate(bounding_box.orientation.inverse)

    corners = bounding_box.corners()

    idx = np.where(
        (corners[0, 7] <= pc[:, 0])
        & (pc[:, 0] <= corners[0, 0])
        & (corners[1, 1] <= pc[:, 1])
        & (pc[:, 1] <= corners[1, 0])
        & (corners[2, 2] <= pc[:, 2])
        & (pc[:, 2] <= corners[2, 0])
    )[0]

    bounding_box.rotate(orientation_backup)
    return idx


def calc_displace_vector(points: np.array, curr_box: Box, next_box: Box):
    """点的位移 = 按盒间 delta 旋转平移后的点减去原点。"""
    assert points.shape[1] == 3, "The input points should have dimension 3."

    curr_box.orientation = curr_box.orientation.normalised
    next_box.orientation = next_box.orientation.normalised

    delta_rotation = curr_box.orientation.inverse * next_box.orientation
    rotated_pc = (delta_rotation.rotation_matrix @ points.T).T
    rotated_curr_center = np.dot(delta_rotation.rotation_matrix, curr_box.center)
    delta_center = next_box.center - rotated_curr_center

    rotated_tranlated_pc = rotated_pc + delta_center
    pc_displace_vectors = rotated_tranlated_pc - points

    return pc_displace_vectors


def voxelize_occupy(pts, voxel_size, extents=None, return_indices=False):
    """二值占据体素化。返回 (h, w, d) 的 0/1 布局，可选非空体素索引。"""
    VOXEL_EMPTY = 0
    VOXEL_FILLED = 1

    if pts.shape[1] < 3 or pts.shape[1] > 4:
        raise ValueError("Points have the wrong shape: {}".format(pts.shape))

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))
        filter_idx = np.where(
            (extents[0, 0] < pts[:, 0])
            & (pts[:, 0] < extents[0, 1])
            & (extents[1, 0] < pts[:, 1])
            & (pts[:, 1] < extents[1, 1])
            & (extents[2, 0] < pts[:, 2])
            & (pts[:, 2] < extents[2, 1])
        )[0]
        pts = pts[filter_idx]

    discrete_pts = np.floor(pts[:, :3] / voxel_size).astype(np.int32)

    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    z_col = discrete_pts[:, 2]
    sorted_order = np.lexsort((z_col, y_col, x_col))
    discrete_pts = discrete_pts[sorted_order]

    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1]))
    )
    _, unique_indices = np.unique(contiguous_array, return_index=True)
    unique_indices.sort()
    voxel_coords = discrete_pts[unique_indices]

    if extents is not None:
        min_voxel_coord = np.floor(extents.T[0] / voxel_size)
        max_voxel_coord = np.ceil(extents.T[1] / voxel_size) - 1
    else:
        min_voxel_coord = np.amin(voxel_coords, axis=0)
        max_voxel_coord = np.amax(voxel_coords, axis=0)

    num_divisions = ((max_voxel_coord - min_voxel_coord) + 1).astype(np.int32)
    voxel_indices = (voxel_coords - min_voxel_coord).astype(int)

    leaf_layout = VOXEL_EMPTY * np.ones(num_divisions.astype(int), dtype=np.float32)
    leaf_layout[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = VOXEL_FILLED

    if return_indices:
        return leaf_layout, voxel_indices
    return leaf_layout


def gen_2d_grid_gt(
    data_dict: dict,
    grid_size: np.array,
    extents: np.array = None,
    frame_skip: int = 0,
    reordered: bool = False,
    proportion_thresh: float = 0.5,
    category_num: int = 5,
    one_hot_thresh: float = 0.8,
    h_flip: bool = False,
    min_point_num_per_voxel: int = -1,
    return_past_2d_disp_gt: bool = False,
    return_instance_map: bool = False,
):
    """生成 2D 网格 GT：占据像素的位移场、类别图、实例图。逻辑与上游逐行一致。"""
    num_sweeps = data_dict["num_sweeps"]
    times = data_dict["times"]
    num_past_sweeps = len(np.where(times >= 0)[0])
    num_future_sweeps = len(np.where(times < 0)[0])
    assert num_past_sweeps + num_future_sweeps == num_sweeps, "The number of sweeps is incorrect!"

    pc_list = []
    for i in range(num_sweeps):
        pc = data_dict["pc_" + str(i)]
        if h_flip:
            pc[0, :] = -pc[0, :]  # flip x coordinate
        pc_list.append(pc.T)

    num_instances = data_dict["num_instances"]
    instance_box_list = list()
    instance_cat_list = list()

    for i in range(num_instances):
        instance = data_dict["instance_boxes_" + str(i)]
        category = data_dict["category_" + str(i)]
        instance_box_list.append(instance)
        instance_cat_list.append(category)

    # -- 过滤并排序参考点云
    refer_pc = pc_list[0]
    refer_pc = refer_pc[:, 0:3]

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))
        filter_idx = np.where(
            (extents[0, 0] < refer_pc[:, 0])
            & (refer_pc[:, 0] < extents[0, 1])
            & (extents[1, 0] < refer_pc[:, 1])
            & (refer_pc[:, 1] < extents[1, 1])
            & (extents[2, 0] < refer_pc[:, 2])
            & (refer_pc[:, 2] < extents[2, 1])
        )[0]
        refer_pc = refer_pc[filter_idx]

    discrete_pts = np.floor(refer_pc[:, 0:2] / grid_size).astype(np.int32)
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    sorted_order = np.lexsort((y_col, x_col))

    refer_pc = refer_pc[sorted_order]
    discrete_pts = discrete_pts[sorted_order]

    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1]))
    )
    _, unique_indices = np.unique(contiguous_array, return_index=True)
    unique_indices.sort()
    pixel_coords = discrete_pts[unique_indices]

    num_points_in_pixel = np.diff(unique_indices)
    num_points_in_pixel = np.append(num_points_in_pixel, discrete_pts.shape[0] - unique_indices[-1])

    if extents is not None:
        min_pixel_coord = np.floor(extents.T[0, 0:2] / grid_size)
        max_pixel_coord = np.ceil(extents.T[1, 0:2] / grid_size) - 1
    else:
        min_pixel_coord = np.amin(pixel_coords, axis=0)
        max_pixel_coord = np.amax(pixel_coords, axis=0)

    num_divisions = ((max_pixel_coord - min_pixel_coord) + 1).astype(np.int32)
    pixel_indices = (pixel_coords - min_pixel_coord).astype(int)

    # -- 点按实例归属
    refer_box_list = list()
    refer_pc_idx_per_bbox = list()
    points_category = np.zeros(refer_pc.shape[0], dtype=int)

    pixel_instance_id = np.zeros(pixel_indices.shape[0], dtype=np.uint8)
    points_instance_id = np.zeros(refer_pc.shape[0], dtype=int)

    for i in range(num_instances):
        instance_cat = instance_cat_list[i]
        instance_box_data = instance_box_list[i][0]
        assert not np.isnan(instance_box_data).any(), "In the keyframe, there should not be NaN box annotation!"

        if h_flip:
            tmp_quad = instance_box_data[6:].copy()
            tmp_quad[2] *= -1
            tmp_quad[3] *= -1
            tmp_quad = Quaternion(tmp_quad)
            tmp_center = instance_box_data[0:3].copy()
            tmp_center[0] = -tmp_center[0]
            tmp_box = Box(center=tmp_center, size=instance_box_data[3:6], orientation=Quaternion(tmp_quad))
        else:
            tmp_box = Box(
                center=instance_box_data[:3], size=instance_box_data[3:6], orientation=Quaternion(instance_box_data[6:])
            )
        idx = point_in_hull_fast(refer_pc[:, 0:3], tmp_box)
        refer_pc_idx_per_bbox.append(idx)
        refer_box_list.append(tmp_box)

        points_category[idx] = instance_cat
        points_instance_id[idx] = i + 1

    assert np.max(points_instance_id) <= 255, "The instance id exceeds uint8 max."

    if len(refer_pc_idx_per_bbox) > 0:
        refer_pc_idx_inside_box = np.concatenate(refer_pc_idx_per_bbox).tolist()
    else:
        refer_pc_idx_inside_box = []
    refer_pc_idx_outside_box = set(range(refer_pc.shape[0])) - set(refer_pc_idx_inside_box)
    refer_pc_idx_outside_box = list(refer_pc_idx_outside_box)

    # -- 像素类别（硬 one-hot 或软概率）
    pixel_cat = np.zeros([unique_indices.shape[0], category_num], dtype=np.float32)
    most_freq_info = []

    for h, v in enumerate(zip(unique_indices, num_points_in_pixel)):
        pixel_elements_categories = points_category[v[0]: v[0] + v[1]]
        elements_freq = np.bincount(pixel_elements_categories, minlength=category_num)
        assert np.sum(elements_freq) == v[1], "The frequency count is incorrect."

        elements_freq = elements_freq / float(v[1])
        most_freq_cat, most_freq = np.argmax(elements_freq), np.max(elements_freq)
        most_freq_info.append([most_freq_cat, most_freq])

        most_freq_elements_idx = np.where(pixel_elements_categories == most_freq_cat)[0]
        pixel_elements_instance_ids = points_instance_id[v[0]: v[0] + v[1]]
        most_freq_instance_id = pixel_elements_instance_ids[most_freq_elements_idx[0]]

        if most_freq >= one_hot_thresh:
            one_hot_cat = np.zeros(category_num, dtype=np.float32)
            one_hot_cat[most_freq_cat] = 1.0
            pixel_cat[h] = one_hot_cat
            pixel_instance_id[h] = most_freq_instance_id
        else:
            pixel_cat[h] = elements_freq

    pixel_cat_map = np.zeros((num_divisions[0], num_divisions[1], category_num), dtype=np.float32)
    pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1]] = pixel_cat[:]

    pixel_instance_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.uint8)
    pixel_instance_map[pixel_indices[:, 0], pixel_indices[:, 1]] = pixel_instance_id[:]

    non_empty_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
    non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

    cell_pts_num = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
    cell_pts_num[pixel_indices[:, 0], pixel_indices[:, 1]] = num_points_in_pixel[:]
    tmp_pixel_cat_map = np.argmax(pixel_cat_map, axis=2)
    ignore_mask = np.logical_and(cell_pts_num <= min_point_num_per_voxel, tmp_pixel_cat_map != 0)
    ignore_mask = np.logical_not(ignore_mask)
    ignore_mask = np.expand_dims(ignore_mask, axis=2)

    # -- 对每个未来帧算位移场
    all_disp_field_gt_list = list()
    all_valid_pixel_maps_list = list()

    past_part = list(range(0, num_past_sweeps, frame_skip + 1))
    future_part = list(range(num_past_sweeps + frame_skip, num_sweeps, frame_skip + 1))
    if return_past_2d_disp_gt:
        zero_disp_field = np.zeros((num_divisions[0], num_divisions[1], 2), dtype=np.float32)
        all_disp_field_gt_list.append(zero_disp_field)
        all_valid_pixel_maps_list.append(non_empty_map)
        frame_considered = np.asarray(past_part + future_part)
        frame_considered = frame_considered[1:]
    else:
        frame_considered = np.asarray(future_part)

    for i in frame_considered:
        curr_disp_vectors = np.zeros_like(refer_pc, dtype=np.float32)
        curr_disp_vectors.fill(np.nan)
        curr_disp_vectors[refer_pc_idx_outside_box,] = 0.0

        for j in range(num_instances):
            instance_box = instance_box_list[j]
            instance_box_data = instance_box[i]

            if np.isnan(instance_box_data).any():
                continue

            if h_flip:
                tmp_quad = instance_box_data[6:].copy()
                tmp_quad[2] *= -1
                tmp_quad[3] *= -1
                tmp_quad = Quaternion(tmp_quad)
                tmp_center = instance_box_data[0:3].copy()
                tmp_center[0] = -tmp_center[0]
                tmp_box = Box(center=tmp_center, size=instance_box_data[3:6], orientation=Quaternion(tmp_quad))
            else:
                tmp_box = Box(
                    center=instance_box_data[:3],
                    size=instance_box_data[3:6],
                    orientation=Quaternion(instance_box_data[6:]),
                )
            pc_in_bbox_idx = refer_pc_idx_per_bbox[j]
            disp_vectors = calc_displace_vector(refer_pc[pc_in_bbox_idx], refer_box_list[j], tmp_box)
            curr_disp_vectors[pc_in_bbox_idx] = disp_vectors[:]

        disp_field = np.zeros([unique_indices.shape[0], 2], dtype=np.float32)
        valid_pixels = np.zeros(unique_indices.shape[0], dtype=bool)

        for h, v in enumerate(zip(unique_indices, num_points_in_pixel)):
            pixel_elements_categories = points_category[v[0]: v[0] + v[1]]
            most_freq_cat, most_freq = most_freq_info[h]

            if most_freq >= proportion_thresh:
                most_freq_cat_idx = np.where(pixel_elements_categories == most_freq_cat)[0]
                most_freq_cat_disp_vectors = curr_disp_vectors[v[0]: v[0] + v[1], :3]
                most_freq_cat_disp_vectors = most_freq_cat_disp_vectors[most_freq_cat_idx]

                if np.isnan(most_freq_cat_disp_vectors).any():
                    valid_pixels[h] = 0.0
                else:
                    mean_disp_vector = np.mean(most_freq_cat_disp_vectors, axis=0)
                    disp_field[h] = mean_disp_vector[0:2]
                    valid_pixels[h] = 1.0

        disp_field_sparse = np.zeros((num_divisions[0], num_divisions[1], 2), dtype=np.float32)
        disp_field_sparse[pixel_indices[:, 0], pixel_indices[:, 1]] = disp_field[:]
        disp_field_sparse = disp_field_sparse * ignore_mask

        valid_pixel_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
        valid_pixel_map[pixel_indices[:, 0], pixel_indices[:, 1]] = valid_pixels[:]

        all_disp_field_gt_list.append(disp_field_sparse)
        all_valid_pixel_maps_list.append(valid_pixel_map)

    all_disp_field_gt_list = np.stack(all_disp_field_gt_list, axis=0)
    all_valid_pixel_maps_list = np.stack(all_valid_pixel_maps_list, axis=0)

    if reordered and return_past_2d_disp_gt:
        num_past = len(past_part)
        all_disp_field_gt_list[0:num_past] = all_disp_field_gt_list[(num_past - 1)::-1]
        all_valid_pixel_maps_list[0:num_past] = all_valid_pixel_maps_list[(num_past - 1)::-1]

    if return_instance_map:
        return (
            all_disp_field_gt_list,
            all_valid_pixel_maps_list,
            non_empty_map,
            pixel_cat_map,
            pixel_indices,
            pixel_instance_map,
        )
    return all_disp_field_gt_list, all_valid_pixel_maps_list, non_empty_map, pixel_cat_map, pixel_indices


def classify_speed_level(all_disp_field_gt, total_future_sweeps=20, future_frame_skip=0):
    """动静二分类：所有评测帧位移 ≤ 0.2 m/s 换算上限即静态。阈值 0.2 与上游一致。"""
    all_disp_field_gt_norm = np.linalg.norm(all_disp_field_gt, ord=2, axis=-1)

    upper_thresh = 0.2
    upper_bound = (future_frame_skip + 1) / 20 * upper_thresh
    selected_future_sweeps = np.arange(0, total_future_sweeps + 1, future_frame_skip + 1)
    selected_future_sweeps = selected_future_sweeps[1:]

    future_sweeps_disp_field_gt_norm = all_disp_field_gt_norm[-len(selected_future_sweeps):, ...]
    static_cell_mask = future_sweeps_disp_field_gt_norm <= upper_bound
    static_cell_mask = np.all(static_cell_mask, axis=0)
    moving_cell_mask = np.logical_not(static_cell_mask)

    motion_cat = np.zeros((all_disp_field_gt.shape[1], all_disp_field_gt.shape[2], 2), dtype=np.float32)
    bg_one_hot_vector = np.zeros(2, dtype=np.float32)
    bg_one_hot_vector[0] = 1.0
    motion_cat[static_cell_mask] = bg_one_hot_vector[:]

    fg_one_hot_vector = np.zeros(2, dtype=np.float32)
    fg_one_hot_vector[1] = 1.0
    motion_cat[moving_cell_mask] = fg_one_hot_vector[:]

    return motion_cat
