# Copyright (C) qiankangan@thu
#
# Tsinghua-Hesai

import argparse
import os,json

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
# from nuscenes.nuscenes import NuScenes
from HeTools.Hedata import CustomDataset, CustomDatasetV2
# from nuscenes.utils.data_classes import Box
from data.dataH_utils import LidarPointCloud,HeLidarPointCloud,Box
from pyquaternion import Quaternion

from data.data_utils import calc_displace_vector, point_in_hull_fast, voxelize_occupy,\
                            calc_displace_vector_He, point_in_hull_fast_He
from model.model import FeatEncoder, MotionNet, MotionNetMGDA

color_map = {0: "c", 1: "m", 2: "k", 3: "y", 4: "r"}
cat_names = {0: "bg", 1: "bus", 2: "ped", 3: "bike", 4: "other"}

os.environ['CUDA_VISIBLE_DEVICES'] = "6"
def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


def vis_scene_data(
    data_path=None,
    index_file_path="/home/qiankangan/MotionNet/HeTools/trainval.list",
    which_scene=0,
    max_seq_num=10,
    begin_frame=0,
    frame_skip=5,
    trained_model_path=None,
    img_save_dir=None,
    which_model="MotionNet",
    use_adj_frame_pred=True,
    use_motion_state_pred_masking=True,
    disp=True,
):
    """
    Visualize the scene data.

    nuscenes_path: the path to the nuScenes dataset
    nuscenes_version: the dataset version ['v1.0-trainval'/'v1.0-mini']
    which_scene: for which we want to visualize
    max_seq_num: how many frames want to visualize
    begin_frame: for this scene, from which frame we want to start our prediction
    frame_skip: how many future frames we want to skip. This is used for generated preprocessed bev data.
    trained_model_path: the path to the pretrained model
    img_save_dir: the directory for saving the predicted image
    which_model: which network ['MotionNet'/'MotionNetMGDA']
    use_adj_frame_pred: whether to predict the relative offsets between two adjacent frames
    use_motion_state_pred_masking: whether to threshold the prediction with motion state estimation results
    disp: whether to immediately show the predicted results
    """
    if data_path is None:
        raise ValueError("Should specify the nuScenes data path.")

    # sample_json_files="./data/sample/train_output_file.json"
    sample_json_files = "./data/wt128_0_sample/wt128_0_train_output_file.json"
    with open(sample_json_files, 'r') as f:
        lines = f.read()
        parsed_data = json.loads(lines)
    
    if args.datavision == "V1":
        hedataset = CustomDataset(root_path=data_path, sensor_type="wt128_0", data_type="train", \
                                split_file_path="/home/qiankangan/MotionNet/data/trainval.list", \
                                frame_skip=5, history_frames = 20, future_frames = 10)
    elif args.datavision == "V2":
        hedataset = CustomDatasetV2(root_path=data_path, sensor_type="wt128_0", data_type="train", data_vision="V2", \
                                split_file_path="/mnt/personal/ts/others/qinghua/shiyining/seqs.list", \
                                save_sample_file_path="./data/sample", path_map_json="/mnt/personal/ts/others/qinghua/shiyining/at2wt.json", \
                                frame_skip=1, history_frames = 20, future_frames = 10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nsweeps_back = 20
    nsweeps_forward = 10 
    past_frame_skip = 0  # when generating the BEV maps, how many history frames need to be skipped
    future_frame_skip = 0  # when generating the BEV maps, how many future frames need to be skipped
    num_frame_skipped = 0

    voxel_size = (0.125, 0.125, 0.4)
    # area_extents = np.array([[-32.0, 32.0], [-32.0, 32.0], [-3, 2]])
    area_extents = np.array([[0, 64.0], [-16., 16.], [-2, 3]])

    sample_cnt = 1
    # class_map = {
    #     "vehicle.car": 1,
    #     "vehicle.bus.rigid": 1,
    #     "vehicle.bus.bendy": 1,
    #     "human.pedestrian": 2,
    #     "vehicle.bicycle": 3,
    # }  # background: 0, other: 4
    class_map = {'bus': 1, 'ped': 2, 'bike':3}
    curr_scene = hedataset.get_samples_in_scenes(which_scene)

    scene_name = curr_scene

    has_reached_last_keyframe = False
    seq_num = 0

    # Load pre-trained network weights
    loaded_models = list()
    if which_model == "MotionNet":
        model = MotionNet(out_seq_len=10, motion_category_num=2, height_feat_size=13)

        model = nn.DataParallel(model)
        checkpoint = torch.load(trained_model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        loaded_models = [model]
    else:
        model_encoder = FeatEncoder()
        model_head = MotionNetMGDA(out_seq_len=20, motion_category_num=2)

        model_encoder = nn.DataParallel(model_encoder)
        model_head = nn.DataParallel(model_head)

        checkpoint = torch.load(trained_model_path)
        model_encoder.load_state_dict(checkpoint["encoder_state_dict"])
        model_head.load_state_dict(checkpoint["head_state_dict"])

        model_encoder = model_encoder.to(device)
        model_head = model_head.to(device)

        loaded_models = [model_encoder, model_head]
    print("Loaded pretrained model {}".format(which_model))


    key_frames = [k for k in parsed_data[scene_name].keys()]


    # iteration : over each timestamp to get the synchronized point clouds
    # print("Processing scene {} | {} ...".format(scene_idx, len(res_scenes)))
    for curr in key_frames:
        cur_timestamp = parsed_data[scene_name][curr]['current_frame']
        curr_timestamp = parsed_data[scene_name][curr]['past_frames'][-1]

        past_timestamps = [t for t in parsed_data[scene_name][curr]['past_frames']]
        future_timestamps = [t for t in parsed_data[scene_name][curr]['future_frames']]

        timestamps = past_timestamps + future_timestamps
        # cur_timestamp = timestamps[nsweeps_back-1]
        # curr_timestamp = float(data[1])
        assert cur_timestamp==curr_timestamp,"The timestamp of the current scene is not correct!"
        assert cur_timestamp==curr,"The timestamp of the current scene is not correct!"
        # past_timestamps = timestamps[0: nsweeps_back]
        # future_timestamps = timestamps[nsweeps_back:]

        assert timestamps[nsweeps_back:] == timestamps[-nsweeps_forward:],"The timestamp is not correct!"


        all_times = np.array(past_timestamps[::-1] + future_timestamps)
        num_sweeps = len(all_times)
        # pc_list = []
        # trans_metrices = []


        # TODO: check all_times?
        all_pc, all_times, trans_metrices = \
            HeLidarPointCloud.from_file_multisweep_bf_sample_data(dataset=hedataset, scene_name=scene_name, \
                                                                timestamps=timestamps,nsweeps_back = nsweeps_back,
                                                                nsweeps_forward = nsweeps_forward,
                                                                return_trans_matrix = True)
        # pc_list.append(all_pc.points)
        # trans_metrices.append(trans_matrix)

        pc = all_pc.points
        _, sort_idx = np.unique(all_times, return_index=True)
        unique_time = all_times[np.sort(sort_idx)]

        # prepare data dictionary for the next step(ie. generating BEV maps)
        save_data_dict = dict()

        # for remmaping the instance ids, according to class map
        save_data_dict = dict()

        curr_token_list = list()

        for tid in range(num_sweeps):
            _time = unique_time[tid]
            points_idx = np.where(all_times==_time)[0]
            _pc = pc[:, points_idx]

            save_data_dict['pc_' + str(tid)] = _pc

        save_data_dict['times'] = unique_time
        save_data_dict['num_sweeps'] = num_sweeps
        save_data_dict['trans_matrices'] = trans_metrices

        # Then get the synchronized annotations

        num_instances = 0 # The number of instance within this sample
        corresponding_sample_token = cur_timestamp

        # TODO: get_annotations_for_timestamp
        # def get_annotations_for_timestamp(scene_name, corresponding_sample_token):
        #     # TODO: implement this function
        #     return []
        corresponding_sample_record = hedataset.get_annotations_for_timestamp(scene_name, corresponding_sample_token)

        for ann in corresponding_sample_record:
            ann_token = ann['ann_token']
            category_name = ann['category_name']
            instance_token = ann['instance_token']

            flag = False
            for c, v in class_map.items():
                if category_name.startswith(c):
                    # save_data_dict["category_%d" % instance_token] = v
                    save_data_dict["category_" + str(num_instances)] = v
                    flag = True
                    break

            if not flag:
                # save_data_dict["category_%d" % instance_token] = 4
                save_data_dict["category_" + str(num_instances)] = 4

            # TODO: check the API
            # def get_instance_boxes_multisweep_sample_data(scene_name, cur_timestamp, instance_token, past_timestamps, future_timestamps):

            instance_boxes, instance_all_times, _, _ = \
                HeLidarPointCloud.get_instance_boxes_multisweep_sample_data(dataset=hedataset, scene_name=scene_name,
                                                        ref_timestamp=timestamps[nsweeps_back - 1],\
                                                        instance_token=instance_token, \
                                                        nsweeps_back=nsweeps_back, nsweeps_forward=nsweeps_forward)
            assert np.array_equal(unique_time, instance_all_times), "The sweep and instance times are inconsistent!"
            assert num_sweeps == len(instance_boxes), "The number of boxes does not match that of sweeps"

            # each row corresponds to a box annotations: center,size,quaternion
            box_data = np.zeros((len(instance_boxes), 7), dtype=np.float32)
            box_data.fill(np.nan)
            for bid, box in enumerate(instance_boxes):
                if box is not None:
                    row = np.concatenate([box["center"], box['lwh'], box['orientation']])
                    box_data[bid, :] = row

            # save the box data for current instance
            # save_data_dict['instance_boxes_%d' % instance_token] = box_data
            save_data_dict["instance_boxes_" + str(num_instances)] = box_data
            num_instances += 1
            curr_token_list.append(instance_token)

        save_data_dict['num_instances'] = num_instances

        if seq_num < begin_frame:
            seq_num += 1
            sample_cnt += 1
            print("Finish loading sequence sample {}".format(seq_num))
            continue

        # ------------------------------------ Visualization ------------------------------------
        # -- The following code is simply borrowed from gen_data.py and currently not optimized
        num_sweeps = save_data_dict["num_sweeps"]
        times = save_data_dict["times"]
        num_past_sweeps = len(np.where(times >= 0)[0])
        num_future_sweeps = len(np.where(times < 0)[0])
        assert num_past_sweeps + num_future_sweeps == num_sweeps, "The number of sweeps is incorrect!"

        # Load point cloud
        pc_list = []

        for i in range(num_sweeps):
            pc = save_data_dict["pc_" + str(i)]
            pc_list.append(pc.T)

        # Reorder the pc, and skip sample frames if wanted
        tmp_pc_list_1 = pc_list[0 : num_past_sweeps : (past_frame_skip + 1)]
        tmp_pc_list_1 = tmp_pc_list_1[::-1]
        tmp_pc_list_2 = pc_list[(num_past_sweeps + future_frame_skip) :: (future_frame_skip + 1)]
        pc_list = tmp_pc_list_1 + tmp_pc_list_2

        num_past_pcs = len(tmp_pc_list_1)
        num_future_pcs = len(tmp_pc_list_2)

        # Voxelize the input point clouds, and compute the ground truth displacement vectors
        padded_voxel_points_list = list()  # This contains the compact representation of voxelization, as in the paper
        padded_pixel_points_list = list()

        for i in range(num_past_pcs):
            res = voxelize_occupy(pc_list[i], voxel_size=voxel_size, extents=area_extents, sensor_type = "wt128_0")
            padded_voxel_points_list.append(res)

        for i in range(num_past_pcs):
            res_pixel = gen_2d_res_pixel(pc_list[i], grid_size=voxel_size[0:2], extents=area_extents)
            padded_pixel_points_list.append(res_pixel)

        first_padded_voxel_points = padded_voxel_points_list[-2]
        last_padded_voxel_points = padded_voxel_points_list[-1]
        res_padded_voxel_points = last_padded_voxel_points - first_padded_voxel_points

        res_voxel_dict = dict()
        res_voxel_dict['first'] = first_padded_voxel_points
        res_voxel_dict['last'] = last_padded_voxel_points
        res_voxel_dict['res'] = res_padded_voxel_points

        res_pixel_dict = dict()
        res_pixel_dict['first'] = padded_pixel_points_list[-2]
        res_pixel_dict['last'] = padded_pixel_points_list[-1]
        res_pixel_dict['res'] = res_pixel_dict['last'] - res_pixel_dict['first']

        res_save_path = "./output"
        # 保存 res_voxel_dict 中的数据
        np.save(os.path.join(res_save_path, 'res_voxel_first.npy'), res_voxel_dict['first'])
        np.save(os.path.join(res_save_path, 'res_voxel_last.npy'), res_voxel_dict['last'])
        np.save(os.path.join(res_save_path, 'res_voxel_res.npy'), res_voxel_dict['res'])

        # 保存 res_pixel_dict 中的数据
        np.save(os.path.join(res_save_path, 'res_pixel_first.npy'), res_pixel_dict['first'])
        np.save(os.path.join(res_save_path, 'res_pixel_last.npy'), res_pixel_dict['last'])
        np.save(os.path.join(res_save_path, 'res_pixel_res.npy'), res_pixel_dict['res'])


        # Compile the batch of voxels, so that they can be fed into the network
        padded_voxel_points = torch.from_numpy(np.stack(padded_voxel_points_list, axis=0))

        # Finally, generate the ground-truth displacement field
        all_disp_field_gt, all_valid_pixel_maps, non_empty_map, pixel_cat_map = gen_2d_grid_gt_for_visualization(
            save_data_dict, grid_size=voxel_size[0:2], reordered=True, extents=area_extents, frame_skip=frame_skip
        )

        bev_input_data = (
            padded_voxel_points,
            res_voxel_dict,
            res_pixel_dict,
            all_disp_field_gt,
            all_valid_pixel_maps,
            non_empty_map,
            pixel_cat_map,
            num_past_pcs,
            num_future_pcs,
        )

        vis_model_per_sample_data(
            bev_input_data,
            save_data_dict,
            frame_skip=frame_skip,
            loaded_models=loaded_models,
            voxel_size=voxel_size,
            which_model=which_model,
            model_path=trained_model_path,
            img_save_dir=img_save_dir,
            use_adj_frame_pred=use_adj_frame_pred,
            disp=disp,
            use_motion_state_pred_masking=use_motion_state_pred_masking,
            frame_idx=seq_num,
        )

        seq_num += 1
        print("Finish loading sequence sample {}".format(seq_num))

        sample_cnt += 1
        # curr_sample_data = nusc.get("sample_data", curr_sample_data["next"])

        if seq_num - begin_frame >= max_seq_num:
            break

def gen_2d_res_pixel(pts, grid_size: np.array, extents: np.array = None,
                                 ):


    # ----------------------------------------------------
    # Filter and sort the reference point cloud
    refer_pc = pts
    refer_pc = refer_pc[:, 0:3]

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))

        filter_idx = np.where((extents[0, 0] < refer_pc[:, 0]) & (refer_pc[:, 0] < extents[0, 1]) &
                              (extents[1, 0] < refer_pc[:, 1]) & (refer_pc[:, 1] < extents[1, 1]) &
                              (extents[2, 0] < refer_pc[:, 2]) & (refer_pc[:, 2] < extents[2, 1]))[0]
        refer_pc = refer_pc[filter_idx]

    # -- Discretize pixel coordinates to given quantization size
    discrete_pts = np.floor(refer_pc[:, 0:2] / grid_size).astype(np.int32)

    # -- Use Lex Sort, sort by x, then y
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    sorted_order = np.lexsort((y_col, x_col))

    refer_pc = refer_pc[sorted_order]
    discrete_pts = discrete_pts[sorted_order]

    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1])))

    # -- The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # -- Sort unique indices to preserve order
    unique_indices.sort()
    pixel_coords = discrete_pts[unique_indices]

    # -- Number of points per voxel, last voxel calculated separately
    num_points_in_pixel = np.diff(unique_indices)
    num_points_in_pixel = np.append(num_points_in_pixel, discrete_pts.shape[0] - unique_indices[-1])

    # -- Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_pixel_coord = np.floor(extents.T[0, 0:2] / grid_size)
        max_pixel_coord = np.ceil(extents.T[1, 0:2] / grid_size) - 1
    else:
        min_pixel_coord = np.amin(pixel_coords, axis=0)
        max_pixel_coord = np.amax(pixel_coords, axis=0)

    # -- Get the voxel grid dimensions
    num_divisions = ((max_pixel_coord - min_pixel_coord) + 1).astype(np.int32)

    # -- Bring the min voxel to the origin
    pixel_indices = (pixel_coords - min_pixel_coord).astype(int)
    non_empty_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
    non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

    return non_empty_map

def gen_2d_grid_gt_for_visualization(
    data_dict: dict,
    grid_size: np.array,
    extents: np.array = None,
    frame_skip: int = 0,
    reordered: bool = False,
    proportion_thresh: float = 0.5,
    category_num: int = 5,
    one_hot_thresh: float = 0.8,
):
    """
    Generate the 2d grid ground-truth for the input point cloud.
    The ground-truth is: the displacement vectors of the occupied pixels in BEV image.
    The displacement is computed w.r.t the current time and the future time

    The difference between this function and gen_2d_grid_gt: the input is "data_dict" instead of sample file path.

    :param data_dict: The dictionary storing point cloud data and annotations
    :param grid_size: The size of each pixel
    :param extents: The extents of the point cloud on the 2D xy plane. Shape (3, 2)
    :param frame_skip: The number of sample frames that need to be skipped
    :param reordered: Whether need to reorder the results, so that the first element corresponds to the oldest record.
    :param proportion_thresh: Within a given pixel, only when the proportion of foreground points exceeds this threshold
        will we compute the displacement vector for this pixel.
    :param category_num: The number of categories for points.
    :param one_hot_thresh: When the proportion of the majority points within a cell exceeds this threshold, we
        compute the (hard) one-hot category vector for this cell, otherwise compute the soft category vector.

    :return: The ground-truth displacement field. Shape (num_sweeps, image height, image width, 2).
    """
    num_sweeps = data_dict["num_sweeps"]
    times = data_dict["times"]
    num_past_sweeps = len(np.where(times >= 0)[0])
    num_future_sweeps = len(np.where(times < 0)[0])
    assert num_past_sweeps + num_future_sweeps == num_sweeps, "The number of sweeps is incorrect!"

    pc_list = []

    for i in range(num_sweeps):
        pc = data_dict["pc_" + str(i)]
        pc_list.append(pc.T)

    # Retrieve the instance boxes
    num_instances = data_dict["num_instances"]
    instance_box_list = list()
    instance_cat_list = list()  # for instance categories

    for i in range(num_instances):
        instance = data_dict["instance_boxes_" + str(i)]
        category = data_dict["category_" + str(i)]
        instance_box_list.append(instance)
        instance_cat_list.append(category)

    # ----------------------------------------------------
    # Filter and sort the reference point cloud
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

    # -- Discretize pixel coordinates to given quantization size
    discrete_pts = np.floor(refer_pc[:, 0:2] / grid_size).astype(np.int32)

    # -- Use Lex Sort, sort by x, then y
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    sorted_order = np.lexsort((y_col, x_col))

    refer_pc = refer_pc[sorted_order]
    discrete_pts = discrete_pts[sorted_order]

    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1]))
    )

    # -- The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # -- Sort unique indices to preserve order
    unique_indices.sort()
    pixel_coords = discrete_pts[unique_indices]

    if unique_indices.shape[0] == 0:
        breakpoint()
    # -- Number of points per voxel, last voxel calculated separately
    num_points_in_pixel = np.diff(unique_indices)
    num_points_in_pixel = np.append(num_points_in_pixel, discrete_pts.shape[0] - unique_indices[-1])

    # -- Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_pixel_coord = np.floor(extents.T[0, 0:2] / grid_size)
        max_pixel_coord = np.ceil(extents.T[1, 0:2] / grid_size) - 1
    else:
        min_pixel_coord = np.amin(pixel_coords, axis=0)
        max_pixel_coord = np.amax(pixel_coords, axis=0)

    # -- Get the voxel grid dimensions
    num_divisions = ((max_pixel_coord - min_pixel_coord) + 1).astype(np.int32)

    # -- Bring the min voxel to the origin
    pixel_indices = (pixel_coords - min_pixel_coord).astype(int)
    # ----------------------------------------------------

    # ----------------------------------------------------
    # Get the point cloud subsets, which are inside different instance bounding boxes
    refer_box_list = list()
    refer_pc_idx_per_bbox = list()
    points_category = np.zeros(refer_pc.shape[0], dtype=int)  # store the point categories

    for i in range(num_instances):
        instance_cat = instance_cat_list[i]
        instance_box = instance_box_list[i]
        instance_box_data = instance_box[0]
        assert not np.isnan(instance_box_data).any(), "In the keyframe, there should not be NaN box annotation!"

        tmp_box = Box(
            center=instance_box_data[:3], size=instance_box_data[3:6], orientation=instance_box_data[6:]
        )
        idx = point_in_hull_fast_He(refer_pc[:, 0:3], tmp_box)
        refer_pc_idx_per_bbox.append(idx)
        refer_box_list.append(tmp_box)

        points_category[idx] = instance_cat

    if len(refer_pc_idx_per_bbox) > 0:
        refer_pc_idx_inside_box = np.concatenate(refer_pc_idx_per_bbox).tolist()
    else:
        refer_pc_idx_inside_box = []
    refer_pc_idx_outside_box = set(range(refer_pc.shape[0])) - set(refer_pc_idx_inside_box)
    refer_pc_idx_outside_box = list(refer_pc_idx_outside_box)

    # Compute pixel (cell) categories
    pixel_cat = np.zeros([unique_indices.shape[0], category_num], dtype=np.float32)
    most_freq_info = []

    for h, v in enumerate(zip(unique_indices, num_points_in_pixel)):
        pixel_elements_categories = points_category[v[0] : v[0] + v[1]]
        elements_freq = np.bincount(pixel_elements_categories, minlength=category_num)
        assert np.sum(elements_freq) == v[1], "The frequency count is incorrect."

        elements_freq = elements_freq / float(v[1])
        most_freq_cat, most_freq = np.argmax(elements_freq), np.max(elements_freq)
        most_freq_info.append([most_freq_cat, most_freq])

        if most_freq >= one_hot_thresh:
            one_hot_cat = np.zeros(category_num, dtype=np.float32)
            one_hot_cat[most_freq_cat] = 1.0
            pixel_cat[h] = one_hot_cat
        else:
            pixel_cat[h] = elements_freq  # we use soft category probability vector.

    pixel_cat_map = np.zeros((num_divisions[0], num_divisions[1], category_num), dtype=np.float32)
    pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1]] = pixel_cat[:]

    # Set the non-zero pixels to 1.0
    # Note that the non-zero pixels correspond to the foreground and background objects
    non_empty_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
    non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

    # Compute the displacement vectors w.r.t. the other sweeps
    all_disp_field_gt_list = list()
    zero_disp_field = np.zeros((num_divisions[0], num_divisions[1], 2), dtype=np.float32)
    all_disp_field_gt_list.append(zero_disp_field)

    all_valid_pixel_maps_list = list()  # valid pixel map will be used for masking the computation of loss
    all_valid_pixel_maps_list.append(non_empty_map)

    # -- Skip some frames if necessary
    past_part = list(range(0, num_past_sweeps, frame_skip + 1))
    future_part = list(range(num_past_sweeps + frame_skip, num_sweeps, frame_skip + 1))
    frame_considered = np.asarray(past_part + future_part)
    # frame_considered = np.asarray(future_part)

    # if:
    #     zero_disp_field = np.zeros((num_divisions[0], num_divisions[1], 2), dtype=np.float32)
    #     all_disp_field_gt_list.append(zero_disp_field)  # append once, which corresponds to the current frame
    #     all_valid_pixel_maps_list.append(non_empty_map)

    #     frame_considered = np.asarray(past_part + future_part)
    #     frame_considered = frame_considered[1:]
    # else:

    for i in frame_considered[1:]:
        curr_disp_vectors = np.zeros_like(refer_pc, dtype=np.float32)
        curr_disp_vectors.fill(np.nan)
        curr_disp_vectors[
            refer_pc_idx_outside_box,
        ] = 0.0

        # First, for each instance, compute the corresponding points displacement.
        for j in range(num_instances):
            instance_box = instance_box_list[j]
            instance_box_data = instance_box[i]  # This is for the i-th sweep

            if np.isnan(instance_box_data).any():  # It is possible that in this sweep there is no annotation
                continue

            tmp_box = Box(
                center=instance_box_data[:3], size=instance_box_data[3:6], orientation=instance_box_data[6:]
            )
            pc_in_bbox_idx = refer_pc_idx_per_bbox[j]
            disp_vectors = calc_displace_vector_He(refer_pc[pc_in_bbox_idx], refer_box_list[j], tmp_box)

            curr_disp_vectors[pc_in_bbox_idx] = disp_vectors[:]

        # Second, compute the mean displacement vector and category for each non-empty pixel
        disp_field = np.zeros([unique_indices.shape[0], 2], dtype=np.float32)  # we only consider the 2D field

        # We only compute loss for valid pixels where there are corresponding box annotations between two frames
        valid_pixels = np.zeros(unique_indices.shape[0], dtype=bool)

        for h, v in enumerate(zip(unique_indices, num_points_in_pixel)):

            # Only when the number of majority points exceeds predefined proportion, we compute
            # the displacement vector for this pixel. Otherwise, We consider it is background (possibly ground plane)
            # and has zero displacement.
            pixel_elements_categories = points_category[v[0] : v[0] + v[1]]
            most_freq_cat, most_freq = most_freq_info[h]

            if most_freq >= proportion_thresh:
                most_freq_cat_idx = np.where(pixel_elements_categories == most_freq_cat)[0]
                most_freq_cat_disp_vectors = curr_disp_vectors[v[0] : v[0] + v[1], :3]
                most_freq_cat_disp_vectors = most_freq_cat_disp_vectors[most_freq_cat_idx]

                if np.isnan(most_freq_cat_disp_vectors).any():  # contains invalid disp vectors
                    valid_pixels[h] = 0.0
                else:
                    mean_disp_vector = np.mean(most_freq_cat_disp_vectors, axis=0)
                    disp_field[h] = mean_disp_vector[0:2]  # ignore the z direction

                    valid_pixels[h] = 1.0

        # Finally, assemble to a 2D image
        disp_field_sparse = np.zeros((num_divisions[0], num_divisions[1], 2), dtype=np.float32)
        disp_field_sparse[pixel_indices[:, 0], pixel_indices[:, 1]] = disp_field[:]

        valid_pixel_map = np.zeros((num_divisions[0], num_divisions[1]), dtype=np.float32)
        valid_pixel_map[pixel_indices[:, 0], pixel_indices[:, 1]] = valid_pixels[:]

        all_disp_field_gt_list.append(disp_field_sparse)
        all_valid_pixel_maps_list.append(valid_pixel_map)

    all_disp_field_gt_list = np.stack(all_disp_field_gt_list, axis=0)
    all_valid_pixel_maps_list = np.stack(all_valid_pixel_maps_list, axis=0)

    if reordered:
        num_past = len(past_part)
        all_disp_field_gt_list[0:num_past] = all_disp_field_gt_list[(num_past - 1) :: -1]
        all_valid_pixel_maps_list[0:num_past] = all_valid_pixel_maps_list[(num_past - 1) :: -1]

    return all_disp_field_gt_list, all_valid_pixel_maps_list, non_empty_map, pixel_cat_map


def vis_model_per_sample_data(bev_input_data, data_dict, frame_skip=3, voxel_size=(0.25, 0.25, 0.4),
                              loaded_models=None, which_model="MotionNet", model_path=None, img_save_dir=None,
                              use_adj_frame_pred=False, use_motion_state_pred_masking=False, frame_idx=0, disp=True):
    """
    Visualize the prediction (ie, displacement field) results.

    bev_ipput_data: the preprocessed sparse bev data
    data_dict: a dictionary storing the point cloud data and annotations
    frame_skip: how many frames we want to skip for future frames
    voxel_size: the size of each voxel
    loaded_models: the model which has loaded the pretrained weights
    which_model: which model to apply ['MotionNet'/'MotionNetMGDA']
    model_path: the path to the pretrained model
    img_save_dir: the directory for saving the predicted image
    use_adj_frame_pred: whether to predict the relative offsets between two adjacent frames
    use_motion_state_pred_masking: whether to threshold the prediction with motion state estimation results
    frame_idx: used for specifying the name of saved image frames
    disp: whether to immediately show the predicted results
    """
    # if model_path is None:
    #     raise ValueError("Need to specify saved model path.")
    if img_save_dir is None:
        raise ValueError("Need to specify image save path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig, ax = plt.subplots(1, 5, figsize=(20, 8))


    # Prepare data for the network
    padded_voxel_points, res_voxel_dict, res_pixel_dict, all_disp_field_gt, all_valid_pixel_maps,\
        non_empty_map, pixel_cat_map_gt, num_past_pcs, num_future_pcs = bev_input_data

    padded_voxel_points = torch.unsqueeze(padded_voxel_points, 0).to(device)

    first_padded_voxel_points = res_voxel_dict['first']
    last_padded_voxel_points = res_voxel_dict['last']
    res_padded_voxel_points = res_voxel_dict['res']
    
    # ------------------------- Visualization -------------------------
    # --- Load the point cloud data and annotations ---
    num_sweeps = data_dict['num_sweeps']
    times = data_dict['times']
    num_past_sweeps = len(np.where(times >= 0)[0])
    num_future_sweeps = len(np.where(times < 0)[0])
    assert num_past_sweeps + num_future_sweeps == num_sweeps, "The number of sweeps is incorrect!"

    # Load point cloud
    pc_list = []

    for i in range(num_sweeps):
        pc = data_dict['pc_' + str(i)]
        pc_list.append(pc)

    # Reorder the pc, and skip sample frames if wanted
    tmp_pc_list_1 = pc_list[0:num_past_sweeps:(frame_skip + 1)]
    tmp_pc_list_1 = tmp_pc_list_1[::-1]
    tmp_pc_list_2 = pc_list[(num_past_sweeps + frame_skip)::(frame_skip + 1)]
    pc_list = tmp_pc_list_1 + tmp_pc_list_2
    num_past_pcs = len(tmp_pc_list_1)

    # Load box annotations, and reorder and skip some annotations if wanted
    num_instances = data_dict['num_instances']
    instance_box_list = list()

    for i in range(num_instances):
        instance = data_dict['instance_boxes_' + str(i)]

        # Reorder the boxes
        tmp_instance = np.zeros((len(pc_list), instance.shape[1]), dtype=np.float32)
        tmp_instance[(num_past_pcs - 1)::-1] = instance[0:num_past_sweeps:(frame_skip + 1)]
        tmp_instance[num_past_pcs:] = instance[(num_past_sweeps + frame_skip)::(frame_skip + 1)]
        instance = tmp_instance[:]

        instance_box_list.append(instance)

    # Draw the LIDAR and quiver plots
    # The distant points are very sparse and not reliable. We do not show them.
    border_meter = 4
    border_pixel = border_meter * 4
    x_lim = [-(32 - border_meter), (32 - border_meter)]
    y_lim = [-(32 - border_meter), (32 - border_meter)]

    # We only show the cells having one-hot category vectors
    max_prob = np.amax(pixel_cat_map_gt, axis=-1)
    filter_mask = max_prob == 1.0
    pixel_cat_map = np.argmax(pixel_cat_map_gt, axis=-1) + 1  # category starts from 1 (background), etc
    pixel_cat_map = (pixel_cat_map * non_empty_map * filter_mask).astype(int)


    # --- Visualization ---
    # --- Visualization ---
    idx = num_past_pcs - 1
    idx_next = idx + 1

    points = pc_list[idx]

    # 定义一个函数来检查点是否在边界框内
    def point_in_box(point, box_center, box_size):
        x_min, y_min, z_min = box_center - box_size / 2
        x_max, y_max, z_max = box_center + box_size / 2
        return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max and z_min <= point[2] <= z_max

    # 获取当前时刻的点云
    points_in_box_idx = []
    for j in range(num_instances):
        inst = instance_box_list[j]

        box_data = inst[idx]
        if np.isnan(box_data).any():
            continue
        for i, point in enumerate(points.T):
            if point_in_box(point, box_data[0:3], box_data[3:6]):
                points_in_box_idx.append(i)
    # breakpoint()
    ax[0].scatter(points[0, :], points[1, :], c=points[2, :], s=1)
    ax[0].set_xlim(x_lim[0], x_lim[1])
    ax[0].set_ylim(y_lim[0], y_lim[1])
    ax[0].axis("off")
    ax[0].set_aspect("equal")
    ax[0].title.set_text("LIDAR data")

    # draw point in box
    in_box_points = points[:, points_in_box_idx]
    if len(in_box_points[0]) > 0:
        ax[0].scatter(in_box_points[0, :], in_box_points[1, :], c='red', s=3)  # 使用红色标记

    for j in range(num_instances):
        inst = instance_box_list[j]

        box_data = inst[idx]
        if np.isnan(box_data).any():
            continue

        box = Box(center=box_data[0:3], size=box_data[3:6], orientation=box_data[6:])
        box.render(ax[0])

    for j in range(num_instances):
        inst = instance_box_list[j]

        box_data = inst[idx_next]
        if np.isnan(box_data).any():
            continue

        box = Box(center=box_data[0:3], size=box_data[3:6], orientation=box_data[6:])
        box.render(ax[0], colors=('g', 'g', 'g'))

    # Plot quiver. We only show non-empty vectors. Plot each category.
    field_gt = all_disp_field_gt[-1]
    idx_x = np.arange(field_gt.shape[0])
    idx_y = np.arange(field_gt.shape[1])
    idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
    qk = [None] * len(color_map)  # for quiver key

    for k in range(len(color_map)):
        # ------------------------ Ground-truth ------------------------
        mask = pixel_cat_map == (k + 1)

        # For cells with very small movements, we threshold them to be static
        field_gt_norm = np.linalg.norm(field_gt, ord=2, axis=-1)  # out: (h, w)
        thd_mask = field_gt_norm <= 0.4
        field_gt[thd_mask, :] = 0

        # Get the displacement field
        X = idx_x[mask]
        Y = idx_y[mask]
        U = field_gt[:, :, 0][mask] / voxel_size[0]  # the distance between pixels is w.r.t. grid size (e.g., 0.2m)
        V = field_gt[:, :, 1][mask] / voxel_size[1]

        qk[k] = ax[1].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color=color_map[k])
        ax[1].quiverkey(qk[k], X=0.0 + k/5.0, Y=1.1, U=20, label=cat_names[k], labelpos='E')
        ax[1].set_xlim(border_pixel, field_gt.shape[0] - border_pixel)
        ax[1].set_ylim(border_pixel, field_gt.shape[1] - border_pixel)
        ax[1].set_aspect('equal')
        ax[1].title.set_text('Ground-truth')
        ax[1].axis('off')

    # plot_voxel_points_3d(res_voxel_dict=res_voxel_dict, savedir=img_save_dir)
    # --------------------for res voxel-------------
    
        
    
    # breakpoint()
    U = np.zeros_like(res_pixel_dict['first']) # the distance between pixels is w.r.t. grid size (e.g., 0.2m)
    V = U
    res_mask = res_pixel_dict['first'] == 1
    qk = ax[2].quiver(idx_x[res_mask], idx_y[res_mask], U[res_mask], V[res_mask], angles='xy', scale_units='xy', scale=1, color=color_map[0])
    ax[2].quiverkey(qk, X=0.0 + k/5.0, Y=1.1, U=20, label=cat_names[k], labelpos='E')
    ax[2].set_xlim(border_pixel, field_gt.shape[0] - border_pixel)
    ax[2].set_ylim(border_pixel, field_gt.shape[1] - border_pixel)
    ax[2].set_aspect('equal')
    ax[2].title.set_text('first')
    ax[2].axis('off')

    res_mask = res_pixel_dict['last'] == 1
    # breakpoint()
    qk = ax[3].quiver(idx_x[res_mask], idx_y[res_mask], U[res_mask], V[res_mask], angles='xy', scale_units='xy', scale=1, color=color_map[0])
    ax[3].quiverkey(qk, X=0.0 + k/5.0, Y=1.1, U=20, label=cat_names[k], labelpos='E')
    ax[3].set_xlim(border_pixel, field_gt.shape[0] - border_pixel)
    ax[3].set_ylim(border_pixel, field_gt.shape[1] - border_pixel)
    ax[3].set_aspect('equal')
    ax[3].title.set_text('last')
    ax[3].axis('off')

    # res_mask = res_pixel_dict['last'] == 1
    # # breakpoint()
    # qk = ax[3].quiver(idx_x[res_mask], idx_y[res_mask], U[res_mask], V[res_mask], angles='xy', scale_units='xy', scale=1, color=color_map[0])
    # ax[3].quiverkey(qk, X=0.0 + k/5.0, Y=1.1, U=20, label=cat_names[k], labelpos='E')
    # ax[3].set_xlim(border_pixel, field_gt.shape[0] - border_pixel)
    # ax[3].set_ylim(border_pixel, field_gt.shape[1] - border_pixel)
    # ax[3].set_aspect('equal')
    # ax[3].title.set_text('last')
    # ax[3].axis('off')

    res_mask = ((res_pixel_dict['last']-res_pixel_dict['first']) == 1)
    res_mask_ = ((res_pixel_dict['last']-res_pixel_dict['first']) == -1)
    # breakpoint()
    qk = ax[4].quiver(idx_x[res_mask], idx_y[res_mask], U[res_mask], V[res_mask], angles='xy', scale_units='xy', scale=1, color=color_map[0])
    ax[4].quiverkey(qk, X=0.0 + k/5.0, Y=1.1, U=20, label=cat_names[k], labelpos='E')
    qk = ax[4].quiver(idx_x[res_mask_], idx_y[res_mask_], U[res_mask_], V[res_mask_], angles='xy', scale_units='xy', scale=1, color=color_map[4])
    ax[4].quiverkey(qk, X=0.0 + k/5.0, Y=1.1, U=20, label=cat_names[k], labelpos='E')
    ax[4].set_xlim(border_pixel, field_gt.shape[0] - border_pixel)
    ax[4].set_ylim(border_pixel, field_gt.shape[1] - border_pixel)
    ax[4].set_aspect('equal')
    ax[4].title.set_text('last')
    ax[4].axis('off')

    # ax[2] = fig.add_subplot(131, projection='3d')
    # # 对于first_padded_voxel_points
    # X, Y, Z = np.indices((256, 256, 13))
    # values = first_padded_voxel_points.flatten()
    # colors = ['blue' if v == 1 else 'white' for v in values]
    # ax[2].scatter(X.flatten(), Y.flatten(), Z.flatten(), c=colors, marker='s', s=10) 
    # ax[2].set_title('First Padded Voxel Points')
    
    
    # # 对于last_padded_voxel_points
    # ax[3] = fig.add_subplot(132, projection='3d')

    # values = last_padded_voxel_points.flatten()
    # colors = ['red' if v == 1 else 'white' for v in values]
    # ax[3].scatter(X.flatten(), Y.flatten(), Z.flatten(), c=colors, marker='s', s=10)
    # ax[3].set_title('Last Padded Voxel Points')

    # # 对于res_padded_voxel_points
    # ax[4] = fig.add_subplot(133, projection='3d')
    # norm = Normalize(vmin=-1, vmax=1)
    # values = res_padded_voxel_points.flatten()
    # colors = plt.cm.RdBu(norm(values))  # 使用RdBu colormap
    # ax[4].scatter(X.flatten(), Y.flatten(), Z.flatten(), c=colors, marker='s', s=10)
    # ax[4].set_title('Res Padded Voxel Points')

    print("finish sample {}".format(frame_idx))
    plt.savefig(os.path.join(img_save_dir, str(frame_idx) + '.png'))

    if disp:
        plt.pause(0.02)
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    ax[3].clear()
    ax[4].clear()

    if disp:
        plt.show()


def gen_scene_prediction_video(images_dir, output_dir, out_format="mp4"):
    images = [
        im for im in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, im)) and im.endswith(".png")
    ]

    num_images = len(images)

    if out_format == "gif":
        save_gif_path = os.path.join(output_dir, "result.gif")
        with imageio.get_writer(save_gif_path, mode="I", fps=20) as writer:
            for i in range(num_images):
                image_file = os.path.join(images_dir, str(i) + ".png")
                image = imageio.imread(image_file)
                writer.append_data(image)

                print("Appending image {}".format(i))
    else:
        save_mp4_path = os.path.join(output_dir, "result.mp4")
        with imageio.get_writer(save_mp4_path, fps=15, quality=10, pixelformat="yuvj444p") as writer:
            for i in range(num_images):
                image_file = os.path.join(images_dir, str(i) + ".png")
                image = imageio.imread(image_file)
                writer.append_data(image)

                print("Appending image {}".format(i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default=None, type=str, help="The path to nuScenes dataset")
    parser.add_argument("-v", "--version", default="v1.0-trainval", type=str, help="The version of nuScenes dataset")
    parser.add_argument("-l", "--savepath", default=None, type=str, help="Directory for saving the generated images")
    parser.add_argument("-n", "--nframe", default=10, type=int, help="The number of frames to be generated")
    parser.add_argument("-s", "--scene", default=5, type=int, help="Which scene")
    parser.add_argument("--datavision", default="V1", type=str, help="Which dataset class")
    parser.add_argument("--net", default="MotionNet", type=str, help="Which network [MotionNet/MotionNetMGDA]")
    parser.add_argument("--modelpath", default=None, type=str, help="Path to the pretrained model")
    parser.add_argument("--beginframe", default=0, type=int, help="From which frame we start predicting")
    parser.add_argument("--format", default="gif", type=str, help="The output animation format [gif/mp4]")

    parser.add_argument("--video", action="store_true", help="Whether to generate images or [gif/mp4]")
    parser.add_argument("--adj", action="store_false", help="Whether predict the relative offset between frames")
    parser.add_argument("--disp", action="store_true", help="Whether to immediately show the images")
    parser.add_argument("--jitter", action="store_false", help="Whether to apply jitter suppression")
    args = parser.parse_args()
    print(args)

    gen_prediction_frames = not args.video
    if_disp = args.disp
    image_save_dir = check_folder(args.savepath)

    # print(f"The processing scene is {scene_list[args.scene]}")
    if gen_prediction_frames:
        if not if_disp:
            matplotlib.use("AGG")

        # index_file_path="/mnt/personal/ts/others/qinghua/shiyining/seqs.list"
        index_file_path = "/home/qiankangan/MotionNet/data/wt128_0_splitfile/wt128_0_train.list"
        with open(index_file_path, 'r') as f:
            scene_list = [line.strip() for line in f.readlines()]
        print(f"The processing scene is {scene_list[args.scene]}")
        vis_scene_data(
            data_path=args.data,
            index_file_path=index_file_path,
            img_save_dir=image_save_dir,
            trained_model_path=args.modelpath,
            which_model=args.net,
            which_scene=scene_list[args.scene],
            frame_skip=4,
            max_seq_num=args.nframe,
            use_adj_frame_pred=args.adj,
            disp=if_disp,
            use_motion_state_pred_masking=args.jitter,
            begin_frame=args.beginframe,
        )
    else:
        frames_dir = os.path.join("/media/pwu/Data/3D_data/nuscene/logs/images_job_talk", image_save_dir)
        save_gif_dir = os.path.join("/media/pwu/Data/3D_data/nuscene/logs/images_job_talk", image_save_dir)
        gen_scene_prediction_video(args.savepath, args.savepath, out_format="gif")
