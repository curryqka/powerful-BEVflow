import os
import numpy as np
import sys
sys.path.append('/workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion')
from multiprocessing import Manager
from utlis.gt_ins_converter import convert_class_to_instance, convert_displacement_to_instance, convert_state_to_instance
from utlis.mask_gt_converterV2 import generate_gt_masks
from data.data_utils import classify_speed_level
class DataLoaderSimple:
    def __init__(
        self,
        dataset_root=None,
        use_velocity_flag=False,
        split="train",
        future_frame_skip=0,
        num_past_frames=5,
        num_future_frames=20,
        voxel_size=(0.25, 0.25, 0.4),
        area_extents=np.array([[-32.0, 32.0], [-32.0, 32.0], [-3.0, 2.0]]),
        num_category=5,
        cache_size=8000,
        sensor_type='at128_0',
        debug_mode=False,
    ):
        if dataset_root is None:
            raise ValueError("The {} dataset root is None. Should specify its value.".format(split))

        self.dataset_root = dataset_root
        print("data root:", dataset_root)

        self.sensor_type = sensor_type
        self.use_velocity_flag = use_velocity_flag

        seq_dirs = []
        if debug_mode:
            for d in os.listdir(self.dataset_root):
                tmp_0 = os.path.join(self.dataset_root, d)[:50] + '/0.npy'
                seq_dirs.append(tmp_0)
        else:
            for d in os.listdir(self.dataset_root):
                tmp_0 = os.path.join(self.dataset_root, d) + '/0.npy'
                seq_dirs.append(tmp_0)

        self.seq_files = seq_dirs

        if debug_mode:
            self.seq_files = self.seq_files[:10]
        self.num_sample_seqs = len(self.seq_files)
        print("The number of {} sequences: {}".format(split, self.num_sample_seqs))

        self.split = split
        self.voxel_size = voxel_size
        self.area_extents = area_extents
        self.num_category = num_category
        self.future_frame_skip = future_frame_skip
        self.num_past_frames = num_past_frames
        manager = Manager()
        self.cache = manager.dict()
        self.cache_size = cache_size if split == "train" else 0

    def load_data(self, idx):
        if idx in self.cache:
            gt_dict = self.cache[idx]
        else:
            seq_file = self.seq_files[idx]
            gt_data_handle = np.load(seq_file, allow_pickle=True)
            gt_dict = gt_data_handle.item()

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

        sparse_pixel_instance_maps = gt_dict['pixel_instance_ids']
        pixel_instance_map = np.zeros((dims[0], dims[1]), dtype=np.uint8)
        pixel_instance_map[pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_pixel_instance_maps[:]

        non_empty_map = np.zeros((dims[0], dims[1]), dtype=np.float32)
        non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0


        # Instance-level GT
        ins_class_single, valid_ins_count = convert_class_to_instance(pixel_cat_map, pixel_instance_map, \
                                                                       max_instances=100, use_batch_dim=False, ignore_id=0, \
                                                                       non_empty_maps=non_empty_map)
        ins_disp_single, valid_ins_disp_count = convert_displacement_to_instance(all_disp_field_gt, pixel_instance_map, \
                                                                                 max_instances=100, use_batch_dim=False, ignore_id=0, \
                                                                                 non_empty_maps=non_empty_map, \
                                                                                 all_disp_valid_maps=all_valid_pixel_maps)
        assert valid_ins_disp_count == valid_ins_count, 'Mismatch valid nums!'

        # Mask
        mask_gt_maps, valid_mask_count = generate_gt_masks(instance_map=pixel_instance_map, ignore_id=0, \
                                                           non_empty_maps=non_empty_map, use_batch_dim=False)
        assert valid_ins_disp_count == valid_mask_count, 'Mismatch valid nums!'

        if self.sensor_type == "wt128_0" and self.use_velocity_flag:
            sparse_pixel_velocity = gt_dict["pixel_velocity"]
            all_pixel_velocity_maps = np.zeros((num_past_pcs, dims[0], dims[1]), dtype=np.float32)
            all_pixel_velocity_maps[:, pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_pixel_velocity

            sparse_pixel_direction = gt_dict["pixel_direction"]
            all_pixel_direction_maps = np.zeros((num_past_pcs, dims[0], dims[1]), dtype=np.float32)
            all_pixel_direction_maps[:, pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_pixel_direction

            sparse_pixel_cos = gt_dict["pixel_cos"]
            all_pixel_cos_maps = np.zeros((num_past_pcs, dims[0], dims[1], 3), dtype=np.float32)
            all_pixel_cos_maps[:, pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_pixel_cos

            all_pixel_velocity_maps = all_pixel_velocity_maps[..., np.newaxis][-self.num_past_frames:]
            all_pixel_direction_maps = all_pixel_direction_maps[..., np.newaxis][-self.num_past_frames:]

        padded_voxel_points = list()
        for i in range(num_past_pcs):
            indices = gt_dict["voxel_indices_" + str(i)]
            curr_voxels = np.zeros(dims, dtype=bool)
            curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
            padded_voxel_points.append(curr_voxels)
        padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)
        padded_voxel_points = padded_voxel_points[-self.num_past_frames:]

        # Classify speed-level (ie, state estimation: static or moving)
        motion_state_gt = classify_speed_level(
            all_disp_field_gt, total_future_sweeps=20, future_frame_skip=self.future_frame_skip
        )
        ins_state_single, _ = convert_state_to_instance(motion_state_gt.transpose(2, 1, 0), pixel_instance_map, \
                                                        max_instances=100, use_batch_dim=False, ignore_id=0, \
                                                        non_empty_maps=non_empty_map)

        return {
            "padded_voxel_points": padded_voxel_points,
            "all_disp_field_gt": all_disp_field_gt,
            "all_valid_pixel_maps": all_valid_pixel_maps,
            "pixel_cat_map": pixel_cat_map,
            "pixel_instance_map": pixel_instance_map,
            "ins_class_single": ins_class_single,
            "ins_disp_single": ins_disp_single,
            "mask_gt_maps": mask_gt_maps,
            "motion_state_gt": motion_state_gt,
            "ins_state_single": ins_state_single,
            "non_empty_map": non_empty_map
        }

def main():
    data_loader = DataLoaderSimple(dataset_root="/SHARE/qiankangan/bevflow/input-data", split="train")
    data = data_loader.load_data(0)
    print("Loaded data keys:", data.keys())

if __name__ == "__main__":
    main()
