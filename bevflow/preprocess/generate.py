# SPDX-License-Identifier: AGPL-3.0-or-later
# 本文件派生自 MERL MotionNet（https://github.com/merlresearch/MotionNet）。
"""nuScenes -> MotionNet 协议预处理数据。从上游 data/gen_data.py 主循环搬运。

与上游的差异（仅结构）：
- 场景列表由调用方传入，支持子集（冒烟/分片），上游只能整跑一个 split。
- 输出目录由调用方给；上游还要求 savepath 父目录已存在（check_folder 非递归），这里 makedirs。
算法逻辑（抽帧、跳帧、实例重排、稀疏化）逐行保持。
"""
import os

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

from bevflow.preprocess.convert import AREA_EXTENTS, VOXEL_SIZE, convert_to_dense_bev, convert_to_sparse_bev

CLASS_MAP = {
    "vehicle.car": 1,
    "vehicle.bus.rigid": 1,
    "vehicle.bus.bendy": 1,
    "human.pedestrian": 2,
    "vehicle.bicycle": 3,
}  # background: 0, other: 4

# train 与 val/test 的抽帧参数不同，来自上游 gen_data.py 顶部分支
SPLIT_PARAMS = {
    "train": {"num_keyframe_skipped": 0, "nsweeps_back": 30, "nsweeps_forward": 20, "skip_frame": 0, "num_adj_seqs": 2},
    "val": {"num_keyframe_skipped": 1, "nsweeps_back": 25, "nsweeps_forward": 20, "skip_frame": 0, "num_adj_seqs": 1},
    "test": {"num_keyframe_skipped": 1, "nsweeps_back": 25, "nsweeps_forward": 20, "skip_frame": 0, "num_adj_seqs": 1},
}


def load_nuscenes(dataroot, version="v1.0-trainval"):
    return NuScenes(version=version, dataroot=dataroot, verbose=False)


def generate_scenes(nusc, scene_ids, split_kind, savepath, log_every=1):
    """对一个场景编号列表跑数据生产。scene_ids 是 nusc.scene 的下标（如 [116, 229]）。

    split_kind 取 train/val/test，决定抽帧参数。输出 <savepath>/<scene_idx>_<seq>/<adj>.npy。
    """
    params = SPLIT_PARAMS[split_kind]
    num_keyframe_skipped = params["num_keyframe_skipped"]
    nsweeps_back = params["nsweeps_back"]
    nsweeps_forward = params["nsweeps_forward"]
    skip_frame = params["skip_frame"]
    num_adj_seqs = params["num_adj_seqs"]

    os.makedirs(savepath, exist_ok=True)
    n_seq_saved = 0

    for scene_idx in scene_ids:
        curr_sample = nusc.get("sample", nusc.scene[scene_idx]["first_sample_token"])
        curr_sample_data = nusc.get("sample_data", curr_sample["data"]["LIDAR_TOP"])

        save_data_dict_list = list()
        save_box_dict_list = list()
        save_instance_token_list = list()
        adj_seq_cnt = 0
        save_seq_cnt = 0

        while curr_sample_data["next"] != "":
            all_pc, all_times, trans_matrices = LidarPointCloud.from_file_multisweep_bf_sample_data(
                nusc,
                curr_sample_data,
                return_trans_matrix=True,
                nsweeps_back=nsweeps_back,
                nsweeps_forward=nsweeps_forward,
            )
            pc = all_pc.points
            _, sort_idx = np.unique(all_times, return_index=True)
            unique_times = all_times[np.sort(sort_idx)]
            num_sweeps = len(unique_times)

            if num_sweeps != (nsweeps_back + nsweeps_forward):
                # 历史/未来 sweep 不足，跳过一些关键帧
                flag = False
                for _ in range(num_keyframe_skipped + 1):
                    if curr_sample["next"] != "":
                        curr_sample = nusc.get("sample", curr_sample["next"])
                    else:
                        flag = True
                        break
                if flag:
                    break
                curr_sample_data = nusc.get("sample_data", curr_sample["data"]["LIDAR_TOP"])
                adj_seq_cnt = 0
                save_data_dict_list = list()
                save_box_dict_list = list()
                save_instance_token_list = list()
                continue

            save_data_dict = dict()
            box_data_dict = dict()
            curr_token_list = list()

            for tid in range(num_sweeps):
                _time = unique_times[tid]
                points_idx = np.where(all_times == _time)[0]
                _pc = pc[:, points_idx]
                save_data_dict["pc_" + str(tid)] = _pc

            save_data_dict["times"] = unique_times
            save_data_dict["num_sweeps"] = num_sweeps
            save_data_dict["trans_matrices"] = trans_matrices

            # 收集该 keyframe 上所有实例的多帧框
            num_instances = 0
            corresponding_sample_token = curr_sample_data["sample_token"]
            corresponding_sample_rec = nusc.get("sample", corresponding_sample_token)

            for ann_token in corresponding_sample_rec["anns"]:
                ann_rec = nusc.get("sample_annotation", ann_token)
                category_name = ann_rec["category_name"]
                instance_token = ann_rec["instance_token"]

                flag = False
                for c, v in CLASS_MAP.items():
                    if category_name.startswith(c):
                        box_data_dict["category_" + instance_token] = v
                        flag = True
                        break
                if not flag:
                    box_data_dict["category_" + instance_token] = 4

                instance_boxes, instance_all_times, _, _ = LidarPointCloud.get_instance_boxes_multisweep_sample_data(
                    nusc, curr_sample_data, instance_token, nsweeps_back=nsweeps_back, nsweeps_forward=nsweeps_forward
                )
                assert np.array_equal(unique_times, instance_all_times), "The sweep and instance times are inconsistent!"
                assert num_sweeps == len(instance_boxes), "The number of instance boxes does not match that of sweeps!"

                box_data = np.zeros((len(instance_boxes), 3 + 3 + 4), dtype=np.float32)
                box_data.fill(np.nan)
                for r, box in enumerate(instance_boxes):
                    if box is not None:
                        row = np.concatenate([box.center, box.wlh, box.orientation.elements])
                        box_data[r] = row[:]

                box_data_dict["instance_boxes_" + instance_token] = box_data
                num_instances += 1
                curr_token_list.append(instance_token)

            save_data_dict["num_instances"] = num_instances
            save_data_dict_list.append(save_data_dict)
            save_box_dict_list.append(box_data_dict)
            save_instance_token_list.append(curr_token_list)

            adj_seq_cnt += 1
            if adj_seq_cnt == num_adj_seqs:
                # 实例 id 重排：公共实例在前，独有实例在后。
                # 与上游的差异：token 按排序遍历。上游用 set 迭代，顺序随进程哈希种子变化，
                # 导致数据生产不可复现（已实验证实：上游重跑自身，重叠同类别框像素的位移场
                # 有厘米级差异）。排序后本管线完全确定；下游一致性损失只依赖同组内标签一致。
                num_instance_token_list = len(save_instance_token_list)
                if num_instance_token_list > 1:
                    common_tokens = set(save_instance_token_list[0]).intersection(save_instance_token_list[1])
                    for l in range(2, num_instance_token_list):
                        common_tokens = common_tokens.intersection(save_instance_token_list[l])

                    for l in range(num_instance_token_list):
                        exclusive_tokens = set(save_instance_token_list[l]).difference(common_tokens)

                        curr_save_data_dict = save_data_dict_list[l]
                        curr_save_box_dict = save_box_dict_list[l]
                        counter = 0
                        for token in sorted(common_tokens):
                            curr_save_data_dict["instance_boxes_" + str(counter)] = curr_save_box_dict[
                                "instance_boxes_" + token
                            ]
                            curr_save_data_dict["category_" + str(counter)] = curr_save_box_dict["category_" + token]
                            counter += 1
                        for token in sorted(exclusive_tokens):
                            curr_save_data_dict["instance_boxes_" + str(counter)] = curr_save_box_dict[
                                "instance_boxes_" + token
                            ]
                            curr_save_data_dict["category_" + str(counter)] = curr_save_box_dict["category_" + token]
                            counter += 1
                        assert counter == curr_save_data_dict["num_instances"], "The number of instances is inconsistent."
                        save_data_dict_list[l] = curr_save_data_dict
                else:
                    curr_save_box_dict = save_box_dict_list[0]
                    curr_save_data_dict = save_data_dict_list[0]
                    for index, token in enumerate(save_instance_token_list[0]):
                        curr_save_data_dict["instance_boxes_" + str(index)] = curr_save_box_dict[
                            "instance_boxes_" + token
                        ]
                        curr_save_data_dict["category_" + str(index)] = curr_save_box_dict["category_" + token]
                    save_data_dict_list[0] = curr_save_data_dict

                for seq_idx, seq_data_dict in enumerate(save_data_dict_list):
                    dense_bev_data = convert_to_dense_bev(seq_data_dict)
                    sparse_bev_data = convert_to_sparse_bev(dense_bev_data)

                    seq_dir = os.path.join(savepath, str(scene_idx) + "_" + str(save_seq_cnt))
                    os.makedirs(seq_dir, exist_ok=True)
                    np.save(os.path.join(seq_dir, str(seq_idx) + ".npy"), arr=sparse_bev_data)

                n_seq_saved += 1
                if n_seq_saved % log_every == 0:
                    print(f"scene {scene_idx}: saved {save_seq_cnt + 1} seqs")

                save_seq_cnt += 1
                adj_seq_cnt = 0
                save_data_dict_list = list()
                save_box_dict_list = list()
                save_instance_token_list = list()

                # 存完一个邻接组后，按 num_keyframe_skipped+1 前进关键帧
                flag = False
                for _ in range(num_keyframe_skipped + 1):
                    if curr_sample["next"] != "":
                        curr_sample = nusc.get("sample", curr_sample["next"])
                    else:
                        flag = True
                        break
                if flag:
                    break
                curr_sample_data = nusc.get("sample_data", curr_sample["data"]["LIDAR_TOP"])
            else:
                # 同组内的相邻序列以 skip_frame+1 个 sample_data 错开（上游行为：错开 1 个 sweep）
                flag = False
                for _ in range(skip_frame + 1):
                    if curr_sample_data["next"] != "":
                        curr_sample_data = nusc.get("sample_data", curr_sample_data["next"])
                    else:
                        flag = True
                        break
                if flag:
                    break

    return n_seq_saved
