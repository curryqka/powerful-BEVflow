import os,json
# from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
import argparse
from data_utils import voxelize_occupy, voxelize_with_velocity, gen_2d_grid_gt, gen_2d_grid_gt_He
from dataH_utils import HeLidarPointCloud
from HeTools.Hedata import CustomDataset, CustomDatasetV2
# def check_folder(folder_name):
#     if not os.path.exists(folder_name):
#         os.mkdir(folder_name)
#     return folder_name

def check_folder(folder_name:str):
    if len(folder_name.split('/')) > 2:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else:
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
        return folder_name

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', default='/media/pwu/Data/3D_data/nuscene/all_nuscene', type=str, help='Root path to nuScenes dataset')
parser.add_argument('-s', '--split', default='train', type=str, help='The data split [train/val/test]')
parser.add_argument('-p', '--savepath', default='/media/pwu/62316788-a8e6-423c-9ed3-303ebb3ab2de/pwu/temporal_data/train', type=str, help='Directory for saving the generated data')
parser.add_argument('-l', '--lidartype', default="at128_0", type=str, help='The lidar type')
parser.add_argument('-v', '--datavision', default="V1", type=str, help='The data vision')
args = parser.parse_args()

# nusc = NuScenes(version='v1.0-trainval', dataroot=args.root, verbose=True)
# print("Total number of scenes:", len(nusc.scene))

# TODO: data split and class_map
# class_map = {'vehicle.car': 1, 'vehicle.bus.rigid': 1, 'vehicle.bus.bendy': 1, 'human.pedestrian': 2,
#              'vehicle.bicycle': 3}  # background: 0, other: 4
# cat_names = {0: "bg", 1: "bus", 2: "ped", 3: "bike", 4: "other"}
class_map = {'bus': 1, 'ped': 2, 'bike':3}
if args.split == 'train':
    num_keyframe_skipped = 0  # The number of keyframes we will skip when dumping the data
    nsweeps_back = 20  # Number of frames back to the history (including the current timestamp)
    nsweeps_forward = 10  # Number of frames into the future (does not include the current timestamp)
    skip_frame = 0  # The number of frames skipped for the adjacent sequence
    num_adj_seqs = 2  # number of adjacent sequences, among which the time gap is \delta t
else:
    num_keyframe_skipped = 1
    nsweeps_back = 20  # Setting this to 30 (for training) or 25 (for testing) allows conducting ablation studies on frame numbers
    nsweeps_forward = 10
    skip_frame = 0
    num_adj_seqs = 1


# The specifications for BEV maps
voxel_size = (0.25, 0.25, 0.4)
# area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
area_extents = np.array([[0., 64.], [-32., 32.], [-2., 3.]])
past_frame_skip = 3  # when generating the BEV maps, how many history frames need to be skipped
future_frame_skip = 0  # when generating the BEV maps, how many future frames need to be skipped
num_past_frames_for_bev_seq = 5  # the number of past frames for BEV map sequence

# get splits
# scenes = np.load('data/split.npy', allow_pickle=True).item().get(args.split)
# split_path = os.path.join('./MotionNet/MotionNet','data/split.npy')
# scenes = np.load(split_path, allow_pickle=True).item().get(args.split)
# print("Split: {}, which contains {} scenes.".format(args.split, len(scenes)))
if args.lidartype == "at128_0":
    # split_file_path = "HeTools/test.list"
    split_file_path="HeTools/total_filter_val_filter_01.list"
    # split_file_path = "/mnt/personal/ts/others/qinghua/shiyining/seqs.list"
else:
    split_file_path = None
if args.datavision == "V1":
    hedataset = CustomDataset(root_path=args.root, sensor_type=args.lidartype, data_type=args.split, \
                            split_file_path=split_file_path, \
                            frame_skip=1, history_frames = nsweeps_back, future_frames = nsweeps_forward)
elif args.datavision == "V2":
    hedataset = CustomDatasetV2(root_path=args.root, sensor_type=args.lidartype, data_type=args.split, data_vision="V2", \
                                split_file_path="/mnt/personal/ts/others/qinghua/shiyining/seqs.list", \
                                save_sample_file_path="./data/sample", path_map_json="/mnt/personal/ts/others/qinghua/shiyining/at2wt.json", \
                                frame_skip=1, history_frames = nsweeps_back, future_frames = nsweeps_forward)

def gen_dataV2(data_split):
    res_scenes = list()
    res_scenes = hedataset.scene_names
    # for s in :
    #     s_id = s.split('_')[1]
    #     res_scenes.append(int(s_id))

    with open(data_split, 'r') as f:
        lines = f.read()
        parsed_data = json.loads(lines)


    for scene_idx, scene_name in enumerate(res_scenes):

        # get current frame in one scene info
        curr_scene = hedataset.get_samples_in_scenes(scene_name=scene_name)

        key_frames = [k for k in parsed_data[scene_name].keys()]

        # for storing consecutive sequences of BEV maps; contains timestampes points,etc
        save_data_dict_list = list()

        # for storing each sequence of box annotations
        save_box_dict_list = list()
        save_instance_token_list = list()

        # only used for save data filename
        adj_seq_cnt = 0
        save_seq_cnt = 0

        # iteration : over each timestamp to get the synchronized point clouds
        print("Processing scene {} | {} ...".format(scene_idx, len(res_scenes)))
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


            # all_times = np.array(past_timestamps[::-1] + future_timestamps)

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
            num_sweeps = len(unique_time)
            # print(unique_time,all_times)
            # prepare data dictionary for the next step(ie. generating BEV maps)
            save_data_dict = dict()

            # for remmaping the instance ids, according to class map
            box_data_dict = dict()

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

            corresponding_sample_record = hedataset.get_annotations_for_timestamp(scene_name, corresponding_sample_token)

            for ann in corresponding_sample_record:
                ann_token = ann['ann_token']
                category_name = ann['category_name']
                instance_token = ann['instance_token']

                flag = False
                for c, v in class_map.items():
                    if category_name.startswith(c):
                        box_data_dict["category_%d" % instance_token] = v
                        flag = True
                        break

                if not flag:
                    box_data_dict["category_%d" % instance_token] = 4

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
                box_data_dict['instance_boxes_%d' % instance_token] = box_data
                num_instances += 1
                curr_token_list.append(instance_token)

            save_data_dict['num_instances'] = num_instances
            save_data_dict_list.append(save_data_dict)
            save_box_dict_list.append(box_data_dict)
            save_instance_token_list.append(curr_token_list)

            # update the counter and save the data if desired
            adj_seq_cnt += 1
            if adj_seq_cnt == num_adj_seqs:
                # First, we need to re-organize the instance tokens(ids)
                num_instance_token_list = len(save_instance_token_list)
                if num_instance_token_list > 1:
                    common_tokens = set(save_instance_token_list[0]).intersection(save_instance_token_list[1])
                    for id in range(2, num_instance_token_list):
                        common_tokens = common_tokens.intersection(save_instance_token_list[id])

                    for id in range(num_instance_token_list):
                        exclusive_tokens = set(save_instance_token_list[id]).difference(common_tokens)

                        # store the common instance first, then store the remaining instances
                        curr_save_data_dict = save_data_dict_list[id]
                        curr_save_box_dict = save_box_dict_list[id]
                        counter=0

                        for token in common_tokens:
                            box_info = curr_save_box_dict['instance_boxes_%d' % token]
                            box_cat = curr_save_box_dict['category_%d' % token]

                            curr_save_data_dict['instance_boxes_' + str(counter)] = box_info
                            curr_save_data_dict['category_' + str(counter)] = box_cat

                            counter+=1

                        for token in exclusive_tokens:
                            box_info = curr_save_box_dict['instance_boxes_%d' % token]
                            box_cat = curr_save_box_dict['category_%d' % token]

                            curr_save_data_dict['instance_boxes_' + str(counter)] = box_info
                            curr_save_data_dict['category_' + str(counter)] = box_cat

                            counter+=1
                        assert counter == curr_save_data_dict["num_instances"],"The number of instances does not match!"

                        save_data_dict_list[id] = curr_save_data_dict
                else:
                    curr_save_box_dict = save_box_dict_list[0]
                    curr_save_data_dict = save_data_dict_list[0]
                    for index, token in enumerate(save_instance_token_list[0]):
                        box_info = curr_save_box_dict['instance_boxes_%d' % token]
                        box_cat = curr_save_box_dict['category_%d' % token]

                        curr_save_data_dict['instance_boxes_' + str(index)] = box_info
                        curr_save_data_dict['category_' + str(index)] = box_cat

                    save_data_dict_list[0] = curr_save_data_dict


                for seq_idx, seq_data_dict in enumerate(save_data_dict_list):
                    dense_bev_data = convert_to_dense_bev(seq_data_dict, args.lidartype)
                    sparse_bev_data = convert_to_sparse_bev(dense_bev_data, args.lidartype)

                    # save data
                    save_directory = check_folder(os.path.join(args.savepath, str(scene_idx) + '_' + str(save_seq_cnt)))
                    save_file_name = os.path.join(save_directory, str(seq_idx) + '.npy')
                    np.save(save_file_name, arr=sparse_bev_data)

                    print(" >> Finish sample: {}, sequence: {}".format(save_seq_cnt, seq_idx))

                save_seq_cnt += 1
                save_data_dict_list = list()
                save_box_dict_list = list()
                save_instance_token_list = list()
                adj_seq_cnt = 0

# ---------------------- Convert the raw data into (dense) BEV maps ----------------------
def convert_to_dense_bev(data_dict, sensor_type='at128_0'):
    num_sweeps = data_dict['num_sweeps']
    times = data_dict['times']
    trans_matrices = data_dict['trans_matrices']

    num_past_sweeps = len(np.where(times >= 0)[0])
    num_future_sweeps = len(np.where(times < 0)[0])
    assert num_past_sweeps + num_future_sweeps == num_sweeps, "The number of sweeps is incorrect!"
    # 确定过去帧和未来帧的数量
    # past_times = times[0:nsweeps_back]
    # future_times = times[nsweeps_back:]

    # num_past_sweeps = len(past_times)
    # num_future_sweeps = len(future_times)
    # assert num_past_sweeps + num_future_sweeps == num_sweeps , "The number of sweeps is incorrect!"

    # Load point cloud
    pc_list = []

    for i in range(num_sweeps):
        pc = data_dict['pc_' + str(i)]
        pc_list.append(pc.T)

    # Reorder the pc, and skip sample frames if wanted
    # Currently the past frames in pc_list are stored in the following order [current, current + 1, current + 2, ...]
    # Therefore, we would like to reorder the frames
    # tmp_pc_list_1 = pc_list[0:num_past_sweeps:(past_frame_skip + 1)]
    # tmp_pc_list_1 = tmp_pc_list_1[::-1]
    # tmp_pc_list_2 = pc_list[num_past_sweeps::(future_frame_skip + 1)]
    # pc_list = tmp_pc_list_1 + tmp_pc_list_2  # now the order is: [past frames -> current frame -> future frames]

    tmp_pc_list_1 = pc_list[0:num_past_sweeps:(past_frame_skip + 1)]
    tmp_pc_list_1 = tmp_pc_list_1[::-1]
    tmp_pc_list_2 = pc_list[(num_past_sweeps + future_frame_skip)::(future_frame_skip + 1)]
    pc_list = tmp_pc_list_1 + tmp_pc_list_2  # now the order is: [past frames -> current frame -> future frames]

    num_past_pcs = len(tmp_pc_list_1)
    num_future_pcs = len(tmp_pc_list_2)

    # Discretize the input point clouds, and compute the ground-truth displacement vectors
    # The following two variables contain the information for the
    # compact representation of binary voxels, as described in the paper
    voxel_indices_list = list()
    padded_voxel_points_list = list()

    past_pcs_idx = list(range(num_past_pcs))
    past_pcs_idx = past_pcs_idx[-num_past_frames_for_bev_seq:]  # we typically use 5 past frames (including the current one)
    for i in past_pcs_idx:
        res, voxel_indices = voxelize_occupy(pc_list[i], voxel_size=voxel_size, extents=area_extents, return_indices=True, \
                                sensor_type=sensor_type)
        voxel_indices_list.append(voxel_indices)
        padded_voxel_points_list.append(res)

    if sensor_type == 'wt128_0':
        padded_voxel_velocity_list = list()
        padded_voxel_direction_list = list()
        padded_voxel_cos_voxel_list = list()
        padded_pixel_velocity_list = list()
        padded_pixel_direction_list = list()
        padded_pixel_cos_list = list()

        for i in past_pcs_idx:
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

        padded_voxel_velocity = np.stack(padded_voxel_velocity_list, axis=0)
        padded_voxel_direction = np.stack(padded_voxel_direction_list, axis=0)
        padded_voxel_cos_voxel = np.stack(padded_voxel_cos_voxel_list, axis=0)
        padded_pixel_velocity = np.stack(padded_pixel_velocity_list, axis=0)
        padded_pixel_direction = np.stack(padded_pixel_direction_list, axis=0)
        padded_pixel_cos = np.stack(padded_pixel_cos_list, axis=0)



    # Compile the batch of voxels, so that they can be fed into the network.
    # Note that, the padded_voxel_points in this script will only be used for sanity check.
    padded_voxel_points = np.stack(padded_voxel_points_list, axis=0).astype(bool)

    # Finally, generate the ground-truth displacement field
    # - all_disp_field_gt: the ground-truth displacement vectors for each grid cell
    # - all_valid_pixel_maps: the masking map for valid pixels, used for loss computation
    # - non_empty_map: the mask which represents the non-empty grid cells, used for loss computation
    # - pixel_cat_map: the map specifying the category for each non-empty grid cell
    # - pixel_indices: the indices of non-empty grid cells, used to generate sparse BEV maps
    # - pixel_instance_map: the map specifying the instance id for each grid cell, used for loss computation
    all_disp_field_gt, all_valid_pixel_maps, non_empty_map, pixel_cat_map, pixel_indices, pixel_instance_map \
        = gen_2d_grid_gt_He(data_dict, grid_size=voxel_size[0:2], extents=area_extents,
                         frame_skip=future_frame_skip, return_instance_map=True)

    if sensor_type == 'wt128_0':
        return voxel_indices_list, padded_voxel_points, pixel_indices, pixel_instance_map, all_disp_field_gt,\
            all_valid_pixel_maps, non_empty_map, pixel_cat_map, num_past_frames_for_bev_seq, num_future_pcs, trans_matrices,\
            padded_voxel_velocity, padded_voxel_direction, padded_voxel_cos_voxel,\
                padded_pixel_velocity, padded_pixel_direction, padded_pixel_cos

    return voxel_indices_list, padded_voxel_points, pixel_indices, pixel_instance_map, all_disp_field_gt,\
        all_valid_pixel_maps, non_empty_map, pixel_cat_map, num_past_frames_for_bev_seq, num_future_pcs, trans_matrices

# ---------------------- Convert the dense BEV data into sparse format ----------------------
# This will significantly reduce the space used for data storage
def convert_to_sparse_bev(dense_bev_data, sensor_type='at128_0'):
    if sensor_type == 'at128_0':
        save_voxel_indices_list, save_voxel_points, save_pixel_indices, save_pixel_instance_maps, \
            save_disp_field_gt, save_valid_pixel_maps, save_non_empty_maps, save_pixel_cat_maps, \
            save_num_past_pcs, save_num_future_pcs, save_trans_matrices = dense_bev_data
    else:
        save_voxel_indices_list, save_voxel_points, save_pixel_indices, save_pixel_instance_maps, \
            save_disp_field_gt, save_valid_pixel_maps, save_non_empty_maps, save_pixel_cat_maps, \
            save_num_past_pcs, save_num_future_pcs, save_trans_matrices, \
                save_voxel_velocity, save_voxel_direction, save_voxel_cos_voxel, \
                    save_pixel_velocity, save_pixel_direction, save_pixel_cos = dense_bev_data

    '''qka 20231204'''
    save_valid_pixel_maps = save_valid_pixel_maps.astype(bool)
    save_voxel_dims = save_voxel_points.shape[1:]
    num_categories = save_pixel_cat_maps.shape[-1]

    sparse_disp_field_gt = save_disp_field_gt[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1], :]
    sparse_valid_pixel_maps = save_valid_pixel_maps[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1]]
    sparse_pixel_cat_maps = save_pixel_cat_maps[save_pixel_indices[:, 0], save_pixel_indices[:, 1]]
    sparse_pixel_instance_maps = save_pixel_instance_maps[save_pixel_indices[:, 0], save_pixel_indices[:, 1]]

    if sensor_type == 'wt128_0':
        sparse_pixel_velocity = save_pixel_velocity[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1]]
        sparse_pixel_direction = save_pixel_direction[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1]]
        sparse_pixel_cos = save_pixel_cos[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1], :]

    save_data_dict = dict()
    for i in range(len(save_voxel_indices_list)):
        save_data_dict['voxel_indices_' + str(i)] = save_voxel_indices_list[i].astype(np.int32)

    save_data_dict['disp_field'] = sparse_disp_field_gt
    save_data_dict['valid_pixel_map'] = sparse_valid_pixel_maps
    save_data_dict['pixel_cat_map'] = sparse_pixel_cat_maps
    save_data_dict['num_past_pcs'] = save_num_past_pcs
    save_data_dict['num_future_pcs'] = save_num_future_pcs
    save_data_dict['trans_matrices'] = save_trans_matrices
    save_data_dict['3d_dimension'] = save_voxel_dims
    save_data_dict['pixel_indices'] = save_pixel_indices
    save_data_dict['pixel_instance_ids'] = sparse_pixel_instance_maps

    if sensor_type == 'wt128_0':
        save_data_dict['pixel_velocity'] = sparse_pixel_velocity
        save_data_dict['pixel_direction'] = sparse_pixel_direction
        save_data_dict['pixel_cos'] = sparse_pixel_cos

    # -------------------------------- Sanity Check --------------------------------
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
        indices = save_data_dict['voxel_indices_' + str(i)]
        curr_voxels = np.zeros(save_voxel_dims, dtype=bool)
        curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        assert np.all(curr_voxels == save_voxel_points[i]), "Error: Mismatch"

    # if sensor_type == 'wt128_0':
    #     test_pixel_velocity = np.zeros((save_num_past_pcs, dims[0], dims[1]), dtype=np.float32)
    #     test_pixel_velocity[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1]] = sparse_pixel_velocity[:]
    #     assert np.all(test_pixel_velocity == save_pixel_velocity), "Error: Mismatch"

    #     test_pixel_direction = np.zeros((save_num_past_pcs, dims[0], dims[1]), dtype=np.float32)
    #     test_pixel_direction[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1]] = sparse_pixel_direction[:]
    #     assert np.all(test_pixel_direction == save_voxel_direction), "Error: Mismatch"

    #     test_pixel_cos = np.zeros((save_num_past_pcs, dims[0], dims[1], 3), dtype=np.float32)
    #     test_pixel_cos[:, save_pixel_indices[:, 0], save_pixel_indices[:, 1], :] = sparse_pixel_cos[:]
    return save_data_dict


if __name__ == "__main__":

    save_sample_file_path = hedataset.save_sample_file_path
    if args.datavision == "V1":
        gen_dataV2(f"{save_sample_file_path}/{args.lidartype}_sample/{args.lidartype}_{args.split}_output_file.json")
    else:
        gen_dataV2(f"{save_sample_file_path}/{args.split}_output_file.json")
