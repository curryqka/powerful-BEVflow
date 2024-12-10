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
from HeTools.Hedata import CustomDataset
# from nuscenes.utils.data_classes import Box
from data.dataH_utils import LidarPointCloud,HeLidarPointCloud,Box
from pyquaternion import Quaternion

from data.data_utils import calc_displace_vector, point_in_hull_fast, voxelize_occupy,\
                            calc_displace_vector_He, point_in_hull_fast_He, voxelize_with_velocity
from model.model import FeatEncoder, MotionNet, MotionNetMGDA
from model.model import GeneralMotionNet, MixGeneralMotionNet, VAE_GeneralMotionNet, LowerGeneralMotionNet

color_map = {0: "c", 1: "m", 2: "k", 3: "y", 4: "r"}
cat_names = {0: "bg", 1: "bus", 2: "ped", 3: "bike", 4: "other"}

os.environ['CUDA_VISIBLE_DEVICES'] = "6"
def check_folder(folder_path):
    if len(folder_path.split('/')) >= 2:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    else:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
    return folder_path


def vis_scene_data(
    data_path=None,
    index_file_path="HeTools/total_filter_val_filter_01.list",
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
    args=None,
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
    
    # sample_json_files="/home/qiankangan/MotionNet/data/output_file.json"
    sample_json_files="/mnt/personal/qiankangan/hadata/wt128_0_sample/wt128_0_val_output_file.json"
    with open(sample_json_files, 'r') as f:
        lines = f.read()
        parsed_data = json.loads(lines)
    
    nsweeps_back = 20
    nsweeps_forward = 10
    out_seq_len = nsweeps_forward
    past_frame_skip = 1  # when generating the BEV maps, how many history frames need to be skipped
    future_frame_skip = 0  # when generating the BEV maps, how many future frames need to be skipped
    num_frame_skipped = 0

    # hedataset = CustomDataset(root_path=data_path, sensor_type="wt128_0", data_type="train",
    #                     split_file_path=index_file_path)
    hedataset = CustomDataset(root_path=data_path, sensor_type='wt128_0', data_type='val', \
                            split_file_path=index_file_path, \
                            save_split_file_path = "/mnt/personal/qiankangan/hadata", save_sample_file_path = '/mnt/personal/qiankangan/hadata', \
                            frame_skip=1, history_frames = nsweeps_back, future_frames = nsweeps_forward)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    voxel_size = (0.25, 0.25, 0.4)
    # area_extents = np.array([[-32.0, 32.0], [-32.0, 32.0], [-3, 2]])
    area_extents = np.array([[0, 64.0], [-32., 32.0], [-2, 3]])

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
    height_feat_size = 13
    if which_model == 'MotionNet' or which_model == 'be-sti':
        # model = GeneralMotionNet(out_seq_len=10, motion_category_num=2, height_feat_size=height_feat_size)
        if args.use_v and args.use_vdir:
            if which_model == 'MotionNet':
                model = GeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 2)
            elif which_model == 'be-sti':
                model = Generalbesti(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 2)
        elif args.use_v and not args.use_vdir:
            if which_model == 'MotionNet':
                model = GeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 1)
            elif which_model == 'be-sti':
                model = Generalbesti(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 1)
        else:
            if which_model == 'MotionNet':
                model = MotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
            elif which_model == 'be-sti':
                model = besti(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        model = nn.DataParallel(model)
        checkpoint = torch.load(trained_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        loaded_models = [model]
    elif which_model == "LowerGeneralMotionNet":
        if args.use_v and args.use_vdir:
            model = LowerGeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 2)
        elif args.use_v and not args.use_vdir:
            model = LowerGeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 1)
        else:
            model = LowerGeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        model = nn.DataParallel(model)
        checkpoint = torch.load(trained_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        loaded_models = [model]
    elif which_model == "MixGeneralMotionNet":
        if args.use_v and args.use_vdir:
            model = MixGeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size,
                                    with_direction=True)
        elif args.use_v and not args.use_vdir:
            model = MixGeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size,
                                    with_direction=False)
        else:
            model = MotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        model = nn.DataParallel(model)
        checkpoint = torch.load(trained_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        loaded_models = [model]
    elif which_model == "UniGenMotionNet":

        # Generative + velocity
        if args.use_v and args.use_vdir:
            model = VAE_GeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 2)
        elif args.use_v and not args.use_vdir:
            model = VAE_GeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 1)
        else:
            model = VAE_GeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        model = nn.DataParallel(model)
        checkpoint = torch.load(trained_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
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
                                                        ref_timestamp=cur_timestamp,\
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

        for i in range(num_past_pcs):
            res = voxelize_occupy(pc_list[i], voxel_size=voxel_size, extents=area_extents, sensor_type=args.sensor_type)
            padded_voxel_points_list.append(res)

        # Compile the batch of voxels, so that they can be fed into the network
        padded_voxel_points = torch.from_numpy(np.stack(padded_voxel_points_list, axis=0))

        if args.sensor_type == 'wt128_0':
            padded_voxel_velocity_list = list()
            padded_voxel_direction_list = list()
            padded_voxel_cos_voxel_list = list()
            padded_pixel_velocity_list = list()
            padded_pixel_direction_list = list()
            padded_pixel_cos_list = list()

            for i in range(num_past_pcs):
                velocity_voxel, direction_voxel, direction_cos_voxel,\
                    velocity_pixel, direction_pixel, direction_cos_pixel \
                        = voxelize_with_velocity(pc_list[i], voxel_size=voxel_size, \
                                                extents=area_extents, return_indices=False)
                padded_voxel_velocity_list.append(velocity_voxel)
                padded_voxel_direction_list.append(direction_voxel)
                padded_voxel_cos_voxel_list.append(direction_cos_voxel)
                padded_pixel_velocity_list.append(velocity_pixel)
                padded_pixel_direction_list.append(direction_pixel)
                padded_pixel_cos_list.append(direction_cos_pixel)

            padded_voxel_velocity = torch.from_numpy(np.stack(padded_voxel_velocity_list, axis=0))
            padded_voxel_direction = torch.from_numpy(np.stack(padded_voxel_direction_list, axis=0))
            padded_voxel_cos_voxel = torch.from_numpy(np.stack(padded_voxel_cos_voxel_list, axis=0))
            padded_pixel_velocity = torch.from_numpy(np.stack(padded_pixel_velocity_list, axis=0))
            padded_pixel_direction = torch.from_numpy(np.stack(padded_pixel_direction_list, axis=0))
            padded_pixel_cos = torch.from_numpy(np.stack(padded_pixel_cos_list, axis=0))

        # Finally, generate the ground-truth displacement field
        all_disp_field_gt, all_valid_pixel_maps, non_empty_map, pixel_cat_map = gen_2d_grid_gt_for_visualization(
            save_data_dict, grid_size=voxel_size[0:2], reordered=True, extents=area_extents, frame_skip=frame_skip
        )
        if args.sensor_type == "wt128_0":
            bev_input_data = (
                padded_voxel_points,
                all_disp_field_gt,
                all_valid_pixel_maps,
                non_empty_map,
                pixel_cat_map,
                num_past_pcs,
                num_future_pcs,
                padded_pixel_velocity,
                padded_pixel_direction,
                padded_pixel_cos,
            )
        else:
            bev_input_data = (
                padded_voxel_points,
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
            input_frame=args.input_frames
        )

        seq_num += 1
        print("Finish loading sequence sample {}".format(seq_num))

        sample_cnt += 1
        # curr_sample_data = nusc.get("sample_data", curr_sample_data["next"])

        if seq_num - begin_frame >= max_seq_num:
            break


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


def vis_model_per_sample_data(
    bev_input_data,
    data_dict,
    frame_skip=3,
    voxel_size=(0.25, 0.25, 0.4),
    loaded_models=None,
    which_model="MotionNet",
    model_path=None,
    img_save_dir=None,
    use_adj_frame_pred=True,
    use_motion_state_pred_masking=True,
    frame_idx=0,
    disp=True,
    input_frame=5,
    disp_error=False,
):
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
    if model_path is None:
        raise ValueError("Need to specify saved model path.")
    if img_save_dir is None:
        raise ValueError("Need to specify image save path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if disp_error:
        fig, ax = plt.subplots(1, 4, figsize=(20, 8))
    else:
        fig, ax = plt.subplots(1, 3, figsize=(20, 8))

    # Load pre-trained network weights
    if which_model == 'MotionNet' or which_model == 'be-sti' or which_model == 'UniGenMotionNet' \
        or which_model == "MixGeneralMotionNet" or which_model == "LowerGeneralMotionNet":
        model = loaded_models[0]
    # if which_model == "MotionNet" or which_mode:
    else:
        model_encoder = loaded_models[0]
        model_head = loaded_models[1]

    # Prepare data for the network
    (
        padded_voxel_points,
        all_disp_field_gt,
        all_valid_pixel_maps,
        non_empty_map,
        pixel_cat_map_gt,
        num_past_pcs,
        num_future_pcs,
        padded_pixel_velocity,
        padded_pixel_direction,
        padded_pixel_cos,
    ) = bev_input_data
    # breakpoint()
    padded_voxel_points = torch.unsqueeze(padded_voxel_points[-input_frame:], 0).to(device)
    all_pixel_velocity_maps = torch.unsqueeze(padded_pixel_velocity[-input_frame:].unsqueeze(-1), 0).to(device)
    all_pixel_direction_maps = torch.unsqueeze(padded_pixel_direction[-input_frame:].unsqueeze(-1), 0).to(device)
    # Make prediction
    if which_model == 'MotionNet' or which_model == 'be-sti' or which_model == 'UniGenMotionNet' \
        or which_model == "MixGeneralMotionNet" or which_model == "LowerGeneralMotionNet":
        model.eval()
    else:
        model_encoder.eval()
        model_head.eval()

    with torch.no_grad():
        if which_model == 'MotionNet' or which_model == 'be-sti' \
            or which_model == "MixGeneralMotionNet" or which_model == "LowerGeneralMotionNet":

            if args.use_v and args.use_vdir:
                disp_pred, cat_pred, motion_pred = model(padded_voxel_points, all_pixel_velocity_maps, all_pixel_direction_maps)
            elif args.use_v and not args.use_vdir:
                disp_pred, cat_pred, motion_pred = model(padded_voxel_points, all_pixel_velocity_maps) 
            else:
                disp_pred, cat_pred, motion_pred = model(padded_voxel_points) 
        # disp_pred, cat_pred, motion_pred = model(padded_voxel_points)
        elif which_model == "UniGenMotionNet":
            if args.use_v and args.use_vdir:
                disp_pred, cat_pred, motion_pred, _ = model(padded_voxel_points, prior_velocity=all_pixel_velocity_maps, prior_direction=all_pixel_direction_maps)
            elif args.use_v and not args.use_vdir:
                disp_pred, cat_pred, motion_pred, _ = model(padded_voxel_points, prior_velocity=all_pixel_velocity_maps) 
            else:
                disp_pred, cat_pred, motion_pred, _ = model(padded_voxel_points)
        else:
            shared_feats = model_encoder(padded_voxel_points)
            disp_pred, cat_pred, motion_pred = model_head(shared_feats)

        disp_pred = disp_pred.cpu().numpy()
        disp_pred = np.transpose(disp_pred, (0, 2, 3, 1))
        cat_pred = np.squeeze(cat_pred.cpu().numpy(), 0)

        if use_adj_frame_pred:  # The prediction are the displacement between adjacent frames
            for c in range(1, disp_pred.shape[0]):
                disp_pred[c, ...] = disp_pred[c, ...] + disp_pred[c - 1, ...]

        if use_motion_state_pred_masking:
            motion_pred_numpy = motion_pred.cpu().numpy()
            motion_pred_numpy = np.argmax(motion_pred_numpy, axis=1)
            motion_mask = motion_pred_numpy == 0

            cat_pred_numpy = np.argmax(cat_pred, axis=0)
            cat_mask = np.logical_and(cat_pred_numpy == 0, non_empty_map == 1)
            cat_mask = np.expand_dims(cat_mask, 0)

            cat_weight_map = np.ones_like(motion_pred_numpy, dtype=np.float32)
            cat_weight_map[motion_mask] = 0.0
            cat_weight_map[cat_mask] = 0.0
            cat_weight_map = cat_weight_map[:, :, :, np.newaxis]  # (1, h, w. 1)

            disp_pred = disp_pred * cat_weight_map

    # ------------------------- Visualization -------------------------
    # --- Load the point cloud data and annotations ---
    num_sweeps = data_dict["num_sweeps"]
    times = data_dict["times"]
    num_past_sweeps = len(np.where(times >= 0)[0])
    num_future_sweeps = len(np.where(times < 0)[0])
    assert num_past_sweeps + num_future_sweeps == num_sweeps, "The number of sweeps is incorrect!"

    # Load point cloud
    pc_list = []

    for i in range(num_sweeps):
        pc = data_dict["pc_" + str(i)]
        pc_list.append(pc)

    # Reorder the pc, and skip sample frames if wanted
    tmp_pc_list_1 = pc_list[0 : num_past_sweeps : (frame_skip + 1)]
    tmp_pc_list_1 = tmp_pc_list_1[::-1]
    tmp_pc_list_2 = pc_list[(num_past_sweeps + frame_skip) :: (frame_skip + 1)]
    pc_list = tmp_pc_list_1 + tmp_pc_list_2
    num_past_pcs = len(tmp_pc_list_1)

    # Load box annotations, and reorder and skip some annotations if wanted
    num_instances = data_dict["num_instances"]
    instance_box_list = list()

    for i in range(num_instances):
        instance = data_dict["instance_boxes_" + str(i)]

        # Reorder the boxes
        tmp_instance = np.zeros((len(pc_list), instance.shape[1]), dtype=np.float32)
        tmp_instance[(num_past_pcs - 1) :: -1] = instance[0 : num_past_sweeps : (frame_skip + 1)]
        tmp_instance[num_past_pcs:] = instance[(num_past_sweeps + frame_skip) :: (frame_skip + 1)]
        instance = tmp_instance[:]

        instance_box_list.append(instance)

    # Draw the LIDAR and quiver plots
    # The distant points are very sparse and not reliable. We do not show them.
    border_meter = 4
    border_pixel = border_meter * 4
    x_lim = [0, 64]
    y_lim = [-32, 32]

    # We only show the cells having one-hot category vectors
    max_prob = np.amax(pixel_cat_map_gt, axis=-1)
    filter_mask = max_prob == 1.0
    pixel_cat_map = np.argmax(pixel_cat_map_gt, axis=-1) + 1  # category starts from 1 (background), etc
    pixel_cat_map = (pixel_cat_map * non_empty_map * filter_mask).astype(int)

    cat_pred = np.argmax(cat_pred, axis=0) + 1
    cat_pred = (cat_pred * non_empty_map * filter_mask).astype(int)

    # --- Visualization ---
    idx = num_past_pcs - 1

    points = pc_list[idx]

    ax[0].scatter(points[0, :], points[1, :], c=points[2, :], s=1)
    ax[0].set_xlim(x_lim[0], x_lim[1])
    ax[0].set_ylim(y_lim[0], y_lim[1])
    ax[0].axis("off")
    ax[0].set_aspect("equal")
    ax[0].title.set_text("LIDAR data")

    for j in range(num_instances):
        inst = instance_box_list[j]

        box_data = inst[idx]
        if np.isnan(box_data).any():
            continue

        box = Box(center=box_data[0:3], size=box_data[3:6], orientation=box_data[6:])
        box.render(ax[0])

    # Plot quiver. We only show non-empty vectors. Plot each category.
    field_gt = all_disp_field_gt[-1]
    idx_x = np.arange(field_gt.shape[0])
    idx_y = np.arange(field_gt.shape[1])
    idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing="ij")
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

        qk[k] = ax[1].quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1, color=color_map[k])
        ax[1].quiverkey(qk[k], X=0.0 + k / 5.0, Y=1.1, U=20, label=cat_names[k], labelpos="E")
        ax[1].set_xlim(border_pixel, field_gt.shape[0] - border_pixel)
        ax[1].set_ylim(border_pixel, field_gt.shape[1] - border_pixel)
        ax[1].set_aspect("equal")
        ax[1].title.set_text("Ground-truth")
        ax[1].axis("off")

        # ------------------------ Prediction ------------------------
        # Show the prediction results. We show the cells corresponding to the non-empty one-hot gt cells.
        mask_pred = cat_pred == (k + 1)
        field_pred = disp_pred[-1]  # Show last prediction, ie., the 20-th frame

        # For cells with very small movements, we threshold them to be static
        field_pred_norm = np.linalg.norm(field_pred, ord=2, axis=-1)  # out: (h, w)
        thd_mask = field_pred_norm <= 0.4
        field_pred[thd_mask, :] = 0

        # We use the same indices as the ground-truth, since we are currently focused on the foreground
        X_pred = idx_x[mask_pred]
        Y_pred = idx_y[mask_pred]
        U_pred = field_pred[:, :, 0][mask_pred] / voxel_size[0]
        V_pred = field_pred[:, :, 1][mask_pred] / voxel_size[1]

        ax[2].quiver(X_pred, Y_pred, U_pred, V_pred, angles="xy", scale_units="xy", scale=1, color=color_map[k])
        ax[2].set_xlim(border_pixel, field_pred.shape[0] - border_pixel)
        ax[2].set_ylim(border_pixel, field_pred.shape[1] - border_pixel)
        ax[2].set_aspect("equal")
        ax[2].title.set_text("Prediction")
        ax[2].axis("off")

        if disp_error:
            # ------------------------ error ------------------------
            # Show the prediction results. We show the cells corresponding to the non-empty one-hot gt cells.


            # We use the same indices as the ground-truth, since we are currently focused on the foreground
            X_error = X
            Y_error = Y
            U_error = (field_pred[:, :, 0][mask] - field_gt[:, :, 0][mask] ) / voxel_size[0]
            V_error = (field_pred[:, :, 1][mask] - field_gt[:, :, 1][mask] ) / voxel_size[1]

            ax[3].quiver(X_error, Y_error, U_error, V_error, angles="xy", scale_units="xy", scale=1, color=color_map[k])
            ax[3].set_xlim(border_pixel, field_pred.shape[0] - border_pixel)
            ax[3].set_ylim(border_pixel, field_pred.shape[1] - border_pixel)
            ax[3].set_aspect("equal")
            ax[3].title.set_text("error")
            ax[3].axis("off")
    print("finish sample {}".format(frame_idx))
    plt.savefig(os.path.join(img_save_dir, str(frame_idx) + ".png"))

    if disp:
        plt.pause(0.02)
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    if disp_error:
        ax[3].clear()

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
    parser.add_argument("--net", default="MotionNet", type=str, help="Which network [MotionNet/MotionNetMGDA]")
    parser.add_argument("--modelpath", default=None, type=str, help="Path to the pretrained model")
    parser.add_argument("--beginframe", default=0, type=int, help="From which frame we start predicting")
    parser.add_argument("--format", default="gif", type=str, help="The output animation format [gif/mp4]")

    parser.add_argument("--video", action="store_true", help="Whether to generate images or [gif/mp4]")
    parser.add_argument("--adj", action="store_false", help="Whether predict the relative offset between frames")
    parser.add_argument("--disp", action="store_true", help="Whether to immediately show the images")
    parser.add_argument("--jitter", action="store_false", help="Whether to apply jitter suppression")

    parser.add_argument("--input_frames", default=5, type=int, help="The number of input frames")
    parser.add_argument('--sensor_type', default='nuscenes', type=str, help='The dataset')
    parser.add_argument('--use_v', action='store_true', help='whether to apply velocity')
    parser.add_argument('--use_vdir', action='store_true', help='whether to apply velocity direction')
    parser.add_argument('--debug', action='store_true', help='Whether to use debug mode')

    args = parser.parse_args()
    print(args)
    
    gen_prediction_frames = not args.video
    if_disp = args.disp
    image_save_dir = check_folder(args.savepath)

    # print(f"The processing scene is {scene_list[args.scene]}")
    if gen_prediction_frames:
        if not if_disp:
            matplotlib.use("AGG")

        # index_file_path="HeTools/total_filter_val_filter_01.list"
        # index_file_path = "HeTools/test.list"
        index_file_path = "/mnt/personal/qiankangan/hadata/wt128_0_splitfile/wt128_0_val.list"
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
            args=args,
        )
    else:
        frames_dir = os.path.join("/media/pwu/Data/3D_data/nuscene/logs/images_job_talk", image_save_dir)
        save_gif_dir = os.path.join("/media/pwu/Data/3D_data/nuscene/logs/images_job_talk", image_save_dir)
        gen_scene_prediction_video(args.savepath, args.savepath, out_format="gif")
