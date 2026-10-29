# SPDX-License-Identifier: AGPL-3.0-or-later
# 本文件派生自 MERL MotionNet（https://github.com/merlresearch/MotionNet）。
"""MotionNet 数据加载器。从上游 data/nuscenes_dataloader.py 原样搬运两个 Dataset。

行为保持：缓存机制、稀疏 dict 展开逻辑、classify_speed_level 调用均不变。
数据卡断言在 data_card 一节实现（experiment-run 阶段接入），不在此处。
"""
import hashlib
import json
import os
import warnings
from multiprocessing import Manager

import numpy as np
from torch.utils.data import Dataset

from bevflow.preprocess.grid import classify_speed_level

VOXEL_SIZE = (0.25, 0.25, 0.4)
AREA_EXTENTS = np.array([[-32.0, 32.0], [-32.0, 32.0], [-3.0, 2.0]])

_LISTING_CACHE = os.path.join(
    os.environ.get("BEVFLOW_EXP", "/media/nas_users/kaneqian/Experiment/bevflow"),
    "_cache", "seq_listings")


def _cached_entries(root, kind):
    """NAS 上 17065 次 stat 很慢，目录列表按 root mtime 缓存（见 artifacts_manifest）。

    kind: "dirs" | "files"。root 的 mtime 变化（增删条目）即失效重扫。
    """
    os.makedirs(_LISTING_CACHE, exist_ok=True)
    key = hashlib.sha1(f"{os.path.realpath(root)}:{kind}".encode()).hexdigest()
    cache_file = os.path.join(_LISTING_CACHE, key + ".json")
    root_mtime = os.stat(root).st_mtime
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            data = json.load(f)
        if data.get("mtime") == root_mtime:
            return data["entries"]
    check = os.path.isdir if kind == "dirs" else os.path.isfile
    entries = sorted(
        os.path.join(root, d) for d in os.listdir(root) if check(os.path.join(root, d))
    )
    with open(cache_file, "w") as f:
        json.dump({"mtime": root_mtime, "entries": entries}, f)
    return entries


class TrainDatasetMultiSeq(Dataset):
    def __init__(
        self,
        dataset_root=None,
        future_frame_skip=0,
        voxel_size=VOXEL_SIZE,
        area_extents=AREA_EXTENTS,
        num_past_frames=5,
        num_future_frames=20,
        num_category=5,
        cache_size=10000,
    ):
        """多序列加载，用于时空一致性损失。dataset_root 下每个子目录是一个序列组。"""
        if dataset_root is None:
            raise ValueError("The dataset root is None. Should specify its value.")

        self.dataset_root = dataset_root
        self.seq_dirs = _cached_entries(dataset_root, "dirs")
        self.num_sample_seqs = len(self.seq_dirs)
        if self.num_sample_seqs != 17065:
            warnings.warn(">> The size of training dataset is not 17065.\n")

        self.future_frame_skip = future_frame_skip
        self.voxel_size = voxel_size
        self.area_extents = area_extents
        self.num_category = num_category
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames

        manager = Manager()
        self.cache = manager.dict()
        self.cache_size = cache_size

    def __len__(self):
        return self.num_sample_seqs

    def __getitem__(self, idx):
        if idx in self.cache:
            gt_dict_list = self.cache[idx]
        else:
            seq_dir = self.seq_dirs[idx]
            gt_file_paths = [
                os.path.join(seq_dir, f) for f in os.listdir(seq_dir) if os.path.isfile(os.path.join(seq_dir, f))
            ]
            gt_dict_list = list()
            for gt_file_path in gt_file_paths:
                gt_dict = np.load(gt_file_path, allow_pickle=True).item()
                gt_dict_list.append(gt_dict)

            if len(self.cache) < self.cache_size:
                self.cache[idx] = gt_dict_list

        padded_voxel_points_list = list()
        all_disp_field_gt_list = list()
        all_valid_pixel_maps_list = list()
        non_empty_map_list = list()
        pixel_cat_map_list = list()
        trans_matrices_list = list()
        pixel_instance_map_list = list()

        for gt_dict in gt_dict_list:
            dims = gt_dict["3d_dimension"]
            num_future_pcs = gt_dict["num_future_pcs"]
            pixel_indices = gt_dict["pixel_indices"]
            trans_matrices = gt_dict["trans_matrices"]

            sparse_disp_field_gt = gt_dict["disp_field"]
            all_disp_field_gt = np.zeros((num_future_pcs, dims[0], dims[1], 2), dtype=np.float32)
            all_disp_field_gt[:, pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_disp_field_gt[:]

            sparse_valid_pixel_maps = gt_dict["valid_pixel_map"]
            all_valid_pixel_maps = np.zeros((num_future_pcs, dims[0], dims[1]), dtype=np.float32)
            all_valid_pixel_maps[:, pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_valid_pixel_maps[:]

            sparse_pixel_cat_maps = gt_dict["pixel_cat_map"]
            pixel_cat_map = np.zeros((dims[0], dims[1], self.num_category), dtype=np.float32)
            pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_pixel_cat_maps[:]

            sparse_pixel_instance_maps = gt_dict["pixel_instance_ids"]
            pixel_instance_map = np.zeros((dims[0], dims[1]), dtype=np.uint8)
            pixel_instance_map[pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_pixel_instance_maps[:]

            non_empty_map = np.zeros((dims[0], dims[1]), dtype=np.float32)
            non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

            padded_voxel_points = list()
            for i in range(self.num_past_frames):
                indices = gt_dict["voxel_indices_" + str(i)]
                curr_voxels = np.zeros(dims, dtype=bool)
                curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
                padded_voxel_points.append(curr_voxels)
            padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)

            padded_voxel_points_list.append(padded_voxel_points[-5:])
            all_disp_field_gt_list.append(all_disp_field_gt)
            all_valid_pixel_maps_list.append(all_valid_pixel_maps)
            non_empty_map_list.append(non_empty_map)
            pixel_cat_map_list.append(pixel_cat_map)
            trans_matrices_list.append(trans_matrices)
            pixel_instance_map_list.append(pixel_instance_map)

        padded_voxel_points_list = np.stack(padded_voxel_points_list, 0)
        all_disp_field_gt_list = np.stack(all_disp_field_gt_list, 0)
        all_valid_pixel_maps_list = np.stack(all_valid_pixel_maps_list, 0)
        non_empty_map_list = np.stack(non_empty_map_list, 0)
        pixel_cat_map_list = np.stack(pixel_cat_map_list, 0)
        trans_matrices_list = np.stack(trans_matrices_list, 0)
        pixel_instance_map_list = np.stack(pixel_instance_map_list, 0)

        adj_seq_num = all_disp_field_gt_list.shape[0]
        motion_cat_list = list()
        for i in range(adj_seq_num):
            tmp_motion_cat = classify_speed_level(
                all_disp_field_gt_list[i], total_future_sweeps=20, future_frame_skip=self.future_frame_skip
            )
            motion_cat_list.append(tmp_motion_cat)
        motion_cat_list = np.stack(motion_cat_list, axis=0)

        return (
            padded_voxel_points_list,
            all_disp_field_gt_list,
            all_valid_pixel_maps_list,
            non_empty_map_list,
            pixel_cat_map_list,
            trans_matrices_list,
            motion_cat_list,
            pixel_instance_map_list,
            self.num_past_frames,
            self.num_future_frames,
        )


class DatasetSingleSeq(Dataset):
    def __init__(
        self,
        dataset_root=None,
        split="train",
        future_frame_skip=0,
        voxel_size=VOXEL_SIZE,
        area_extents=AREA_EXTENTS,
        num_category=5,
        cache_size=8000,
        return_instance=False,
    ):
        """单序列加载，评测用。dataset_root 是扁平目录，直接放 .npy 文件（见 scripts/flatten_eval_data.py）。"""
        if dataset_root is None:
            raise ValueError("The {} dataset root is None. Should specify its value.".format(split))

        self.dataset_root = dataset_root
        self.seq_files = _cached_entries(dataset_root, "files")
        self.num_sample_seqs = len(self.seq_files)
        print("The number of {} sequences: {}".format(split, self.num_sample_seqs))

        if split == "train" and self.num_sample_seqs != 17065 * 2:
            warnings.warn(">> The size of training dataset is not 17065 * 2.\n")
        elif split == "val" and self.num_sample_seqs != 1719:
            warnings.warn(">> The size of validation dataset is not 1719.\n")
        elif split == "test" and self.num_sample_seqs != 4309:
            warnings.warn(">> The size of testing dataset is not 4309.\n")

        self.split = split
        self.voxel_size = voxel_size
        self.area_extents = area_extents
        self.num_category = num_category
        self.future_frame_skip = future_frame_skip
        self.return_instance = return_instance

        manager = Manager()
        self.cache = manager.dict()
        self.cache_size = cache_size if split == "train" else 0

    def __len__(self):
        return self.num_sample_seqs

    def __getitem__(self, idx):
        if idx in self.cache:
            gt_dict = self.cache[idx]
        else:
            seq_file = self.seq_files[idx]
            gt_dict = np.load(seq_file, allow_pickle=True).item()
            if len(self.cache) < self.cache_size:
                self.cache[idx] = gt_dict

        dims = gt_dict["3d_dimension"]
        num_future_pcs = gt_dict["num_future_pcs"]
        num_past_pcs = gt_dict["num_past_pcs"]
        pixel_indices = gt_dict["pixel_indices"]

        sparse_disp_field_gt = gt_dict["disp_field"]
        all_disp_field_gt = np.zeros((num_future_pcs, dims[0], dims[1], 2), dtype=np.float32)
        all_disp_field_gt[:, pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_disp_field_gt[:]

        sparse_valid_pixel_maps = gt_dict["valid_pixel_map"]
        all_valid_pixel_maps = np.zeros((num_future_pcs, dims[0], dims[1]), dtype=np.float32)
        all_valid_pixel_maps[:, pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_valid_pixel_maps[:]

        sparse_pixel_cat_maps = gt_dict["pixel_cat_map"]
        pixel_cat_map = np.zeros((dims[0], dims[1], self.num_category), dtype=np.float32)
        pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_pixel_cat_maps[:]

        non_empty_map = np.zeros((dims[0], dims[1]), dtype=np.float32)
        non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

        padded_voxel_points = list()
        for i in range(num_past_pcs):
            indices = gt_dict["voxel_indices_" + str(i)]
            curr_voxels = np.zeros(dims, dtype=bool)
            curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            padded_voxel_points.append(curr_voxels)
        padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)

        motion_state_gt = classify_speed_level(
            all_disp_field_gt, total_future_sweeps=20, future_frame_skip=self.future_frame_skip
        )

        out = (
            padded_voxel_points,
            all_disp_field_gt,
            all_valid_pixel_maps,
            non_empty_map,
            pixel_cat_map,
            num_past_pcs,
            num_future_pcs,
            motion_state_gt,
        )
        if not self.return_instance:
            return out
        # 扩展口径（稳定性指标）需要实例图：稀疏 dict → 稠密
        pixel_instance_map = np.zeros((dims[0], dims[1]), dtype=np.uint8)
        pixel_instance_map[pixel_indices[:, 0], pixel_indices[:, 1]] = gt_dict["pixel_instance_ids"]
        return out + (pixel_instance_map,)
