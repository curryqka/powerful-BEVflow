"""稠密 BEV 组装与稀疏化。从上游 data/gen_data.py 原样搬运，含自校验断言。"""
import numpy as np

from bevflow.preprocess.grid import gen_2d_grid_gt, voxelize_occupy

# 与上游一致的全局 BEV 常量
VOXEL_SIZE = (0.25, 0.25, 0.4)
AREA_EXTENTS = np.array([[-32.0, 32.0], [-32.0, 32.0], [-3.0, 2.0]])
PAST_FRAME_SKIP = 3
FUTURE_FRAME_SKIP = 0
NUM_PAST_FRAMES_FOR_BEV_SEQ = 5


def convert_to_dense_bev(data_dict):
    num_sweeps = data_dict["num_sweeps"]
    times = data_dict["times"]
    trans_matrices = data_dict["trans_matrices"]

    num_past_sweeps = len(np.where(times >= 0)[0])
    num_future_sweeps = len(np.where(times < 0)[0])
    assert num_past_sweeps + num_future_sweeps == num_sweeps, "The number of sweeps is incorrect!"

    pc_list = []
    for i in range(num_sweeps):
        pc = data_dict["pc_" + str(i)]
        pc_list.append(pc.T)

    # 重排：[过去帧 -> 当前帧 -> 未来帧]，按 skip 抽帧
    tmp_pc_list_1 = pc_list[0:num_past_sweeps:(PAST_FRAME_SKIP + 1)]
    tmp_pc_list_1 = tmp_pc_list_1[::-1]
    tmp_pc_list_2 = pc_list[(num_past_sweeps + FUTURE_FRAME_SKIP)::(FUTURE_FRAME_SKIP + 1)]
    pc_list = tmp_pc_list_1 + tmp_pc_list_2

    num_past_pcs = len(tmp_pc_list_1)
    num_future_pcs = len(tmp_pc_list_2)

    voxel_indices_list = list()
    padded_voxel_points_list = list()

    past_pcs_idx = list(range(num_past_pcs))
    past_pcs_idx = past_pcs_idx[-NUM_PAST_FRAMES_FOR_BEV_SEQ:]
    for i in past_pcs_idx:
        res, voxel_indices = voxelize_occupy(
            pc_list[i], voxel_size=VOXEL_SIZE, extents=AREA_EXTENTS, return_indices=True
        )
        voxel_indices_list.append(voxel_indices)
        padded_voxel_points_list.append(res)

    padded_voxel_points = np.stack(padded_voxel_points_list, axis=0).astype(bool)

    (
        all_disp_field_gt,
        all_valid_pixel_maps,
        non_empty_map,
        pixel_cat_map,
        pixel_indices,
        pixel_instance_map,
    ) = gen_2d_grid_gt(
        data_dict,
        grid_size=VOXEL_SIZE[0:2],
        extents=AREA_EXTENTS,
        frame_skip=FUTURE_FRAME_SKIP,
        return_instance_map=True,
    )

    return (
        voxel_indices_list,
        padded_voxel_points,
        pixel_indices,
        pixel_instance_map,
        all_disp_field_gt,
        all_valid_pixel_maps,
        non_empty_map,
        pixel_cat_map,
        NUM_PAST_FRAMES_FOR_BEV_SEQ,
        num_future_pcs,
        trans_matrices,
    )


def convert_to_sparse_bev(dense_bev_data):
    (
        save_voxel_indices_list,
        save_voxel_points,
        save_pixel_indices,
        save_pixel_instance_maps,
        save_disp_field_gt,
        save_valid_pixel_maps,
        save_non_empty_maps,
        save_pixel_cat_maps,
        save_num_past_pcs,
        save_num_future_pcs,
        save_trans_matrices,
    ) = dense_bev_data

    save_valid_pixel_maps = save_valid_pixel_maps.astype(bool)
    save_voxel_dims = save_voxel_points.shape[1:]
    num_categories = save_pixel_cat_maps.shape[-1]

    sparse_disp_field_gt = save_disp_field_gt[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1], :]
    sparse_valid_pixel_maps = save_valid_pixel_maps[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1]]
    sparse_pixel_cat_maps = save_pixel_cat_maps[save_pixel_indices[:, 0], save_pixel_indices[:, 1]]
    sparse_pixel_instance_maps = save_pixel_instance_maps[save_pixel_indices[:, 0], save_pixel_indices[:, 1]]

    save_data_dict = dict()
    for i in range(len(save_voxel_indices_list)):
        save_data_dict["voxel_indices_" + str(i)] = save_voxel_indices_list[i].astype(np.int32)

    save_data_dict["disp_field"] = sparse_disp_field_gt
    save_data_dict["valid_pixel_map"] = sparse_valid_pixel_maps
    save_data_dict["pixel_cat_map"] = sparse_pixel_cat_maps
    save_data_dict["num_past_pcs"] = save_num_past_pcs
    save_data_dict["num_future_pcs"] = save_num_future_pcs
    save_data_dict["trans_matrices"] = save_trans_matrices
    save_data_dict["3d_dimension"] = save_voxel_dims
    save_data_dict["pixel_indices"] = save_pixel_indices
    save_data_dict["pixel_instance_ids"] = sparse_pixel_instance_maps

    # -- Sanity Check：稀疏化与稠密互逆
    dims = save_non_empty_maps.shape

    test_disp_field_gt = np.zeros((save_num_future_pcs, dims[0], dims[1], 2), dtype=np.float32)
    test_disp_field_gt[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1], :] = sparse_disp_field_gt[:]
    assert np.all(test_disp_field_gt == save_disp_field_gt), "Error: Mismatch"

    test_valid_pixel_maps = np.zeros((save_num_future_pcs, dims[0], dims[1]), dtype=bool)
    test_valid_pixel_maps[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1]] = sparse_valid_pixel_maps[:]
    assert np.all(test_valid_pixel_maps == save_valid_pixel_maps), "Error: Mismatch"

    test_pixel_cat_maps = np.zeros((dims[0], dims[1], num_categories), dtype=np.float32)
    test_pixel_cat_maps[save_pixel_indices[:, 0], save_pixel_indices[:, 1], :] = sparse_pixel_cat_maps[:]
    assert np.all(test_pixel_cat_maps == save_pixel_cat_maps), "Error: Mismatch"

    test_non_empty_map = np.zeros((dims[0], dims[1]), dtype=np.float32)
    test_non_empty_map[save_pixel_indices[:, 0], save_pixel_indices[:, 1]] = 1.0
    assert np.all(test_non_empty_map == save_non_empty_maps), "Error: Mismatch"

    test_pixel_instance_map = np.zeros((dims[0], dims[1]), dtype=np.uint8)
    test_pixel_instance_map[save_pixel_indices[:, 0], save_pixel_indices[:, 1]] = sparse_pixel_instance_maps[:]
    assert np.all(test_pixel_instance_map == save_pixel_instance_maps), "Error: Mismatch"

    for i in range(len(save_voxel_indices_list)):
        indices = save_data_dict["voxel_indices_" + str(i)]
        curr_voxels = np.zeros(save_voxel_dims, dtype=bool)
        curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        assert np.all(curr_voxels == save_voxel_points[i]), "Error: Mismatch"

    return save_data_dict
