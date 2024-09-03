import pickle,os,json,random
import numpy as np
from typing import List, Dict, Tuple, Any
from .He_utils import rotation_matrix_to_quat
class CustomDataset:

    def __init__(self, root_path:str, sensor_type:str = 'at128_0', data_type:str = "train", 
                split_file_path:str = None, \
                save_split_file_path: str = "./HeTools", \
                    frame_skip: int = 1, history_frames: int = 5, future_frames: int = 5):
        self.root_path = root_path
        self.sensor_type = sensor_type
        # with open(pkl_file_path, 'rb') as f:
        #     self.data = pickle.load(f)
        self.data_type = data_type
        self.raw_data_path = f"{self.root_path}/raw_data/{sensor_type}"
        self.save_split_file_path = save_split_file_path
        self.frame_skip = frame_skip
        self.history_frames = history_frames
        self.future_frames = future_frames

        # cat_names = {0: "bg", 1: "bus", 2: "ped", 3: "bike", 4: "other"}
        self.cls_map_raw = {

            "Car": "Car",
            "Pedestrian": "Pedestrian",
            "Motorcyclist": "Cyclist",
            "Cyclist": "Cyclist",
            "Tricyclist": "Cyclist",
            "Bicyclist": "Cyclist",
            "Van": "Van",
            "Bus": "Van",
            "Truck": "Van",
            "Trucks": "Van",
            "Special_vehicle": "Van",
            "Person_sitting": "Pedestrian",
            "Misc": "Van",
            "Mini_truck": "Van",
            "Traffic_cone": "Traffic_cone",
            "Construction_sign":"Construction_sign",
            "Water_horse": "Water_horse",
            "Small_movable": "Small_movable",
            "Small_unmovable": "Small_movable",
            "Noise": "Noise",
            "Bicycle":"Bicycle",
            "Tricycle":"Tricycle",
            "Other":"Other"
        }

        self.cls_map = {
            "Car": "bus",
            "Pedestrian": "ped",
            "Motorcyclist": "ped",
            "Cyclist": "ped",
            "Tricyclist": "bike",
            "Bicyclist": "bike",
            "Van": "other",
            "Bus": "bus",
            "Truck": "other",
            "Trucks": "other",
            "Special_vehicle": "other",
            "Person_sitting": "ped",
            "Misc": "other",
            "Mini_truck": "other",
            "Traffic_cone": "other",
            "Construction_sign":"other",
            "Water_horse": "other",
            "Small_movable": "other",
            "Small_unmovable": "other",
            "Noise": "other",
            "Bicycle":"bike",
            "Tricycle":"bike",
            "Other":"other"
        }

        # self.cls_map = {
        #     "Car": "other",
        #     "Pedestrian": "other",
        #     "Motorcyclist": "other",
        #     "Cyclist": "other",
        #     "Tricyclist": "other",
        #     "Bicyclist": "other",
        #     "Van": "other",
        #     "Bus": "other",
        #     "Truck": "other",
        #     "Trucks": "other",
        #     "Special_vehicle": "other",
        #     "Person_sitting": "other",
        #     "Misc": "other",
        #     "Mini_truck": "other",
        #     "Traffic_cone": "other",
        #     "Construction_sign":"other",
        #     "Water_horse": "other",
        #     "Small_movable": "other",
        #     "Small_unmovable": "other",
        #     "Noise": "other",
        #     "Bicycle":"other",
        #     "Tricycle":"other",
        #     "Other":"other"
        # }

        # TODO: revised the index file path
        # split_file_path = "HeTools/total_filter_val_filter_01.list"
        # split_file_path = "HeTools/test.list"
        if split_file_path is None:
            pkl_file_path = f"{self.root_path}/groundtruth/{self.sensor_type}/Det_data_infos"
            split_file_path = self._create_split_file(pkl_file_path)

        train_split_file, val_split_file, test_split_file = self._split_split_file(split_file_path)
        # self.scene_names = self._get_all_scene_names(split_file_path)
        if data_type == "train":
            # split_file_path = None
            with open(train_split_file, 'r') as f:
                self.scene_names = [line.strip() for line in f.readlines()]
        elif data_type == "val":
            with open(val_split_file, 'r') as f:
                self.scene_names = [line.strip() for line in f.readlines()]
        elif data_type == "test":
            with open(test_split_file, 'r') as f:
                self.scene_names = [line.strip() for line in f.readlines()]
        else:
            raise ValueError("data_type must be 'train' or 'val'")
        

        save_sample_files = f"./data/{self.sensor_type}_{self.data_type}_output_file.txt"
        if not os.path.exists(save_sample_files):
            print(f"formating samples:{save_sample_files}")
            self.samples = self._generate_samples()
            # self.data = self._load_data()
            # self.data = self.read_pkl_file()
            self.save_samples_to_file(save_sample_files)

        save_sample_json_files = f"./data/{self.sensor_type}_{self.data_type}_output_file.json"

        if not os.path.exists(save_sample_json_files):
            print(f"formatting samples to json format:{save_sample_json_files}")
            self.samples = self._generate_samples_json()
            self.save_samples_to_jsonfile(save_sample_json_files)

        

    def _create_split_file(self, pkl_file_path: str) -> str:
        # read the split file path to get all scenes to annotations
        save_split_file_path = f"{self.save_split_file_path}/trainval.list"
        with open(save_split_file_path, 'w') as f:
            for scene_name in os.listdir(pkl_file_path):
                f.write(scene_name.split('.')[0] + '\n')

        print("Split file created successfully.")
        return save_split_file_path

    def _split_split_file(self, split_file_path: str, train_ratio=10, val_ratio=3, test_ratio=4) -> None:
        with open(split_file_path, 'r') as file:
            lines = file.readlines()

        lines = [line.strip() for line in lines]
        random.shuffle(lines)

        total_ratio = train_ratio + val_ratio + test_ratio
        train_size = (train_ratio/ total_ratio) * len(lines)
        val_size = (val_ratio / total_ratio) * len(lines)

        train_set = lines[:int(train_size)]
        val_set = lines[int(train_size):int(train_size + val_size)]
        test_set = lines[int(train_size + val_size):]

        set_list = {'train': train_set, 'val': val_set, 'test': test_set}
        
        file_list = []
        
        for name, data in set_list.items():
            if not os.path.exists(f'{self.save_split_file_path}/{self.sensor_type}'):
                os.makedirs(f'{self.save_split_file_path}/{self.sensor_type}')

            
            new_split_file = f'{self.save_split_file_path}/{self.sensor_type}/{self.sensor_type}_{name}.list'
            file_list.append(new_split_file)
            if not os.path.exists(new_split_file):
                with open(new_split_file, 'w') as f:
                    f.write("\n".join(data))
        
        return file_list[0], file_list[1], file_list[2]

    def _get_all_scene_names(self, index_file_path: str) -> List[str]:
        # 读取索引文件，获取所有场景名
        all_scene_names = []
        with open(index_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                scene_name = line.strip()
                pkl_file_path = os.path.join(self.root_path, 'groundtruth', self.sensor_type, 'Det_data_infos', f"{scene_name}.pkl")
                if os.path.exists(pkl_file_path):
                    all_scene_names.append(scene_name)
        return all_scene_names

    def _generate_samples(self) -> List[str]:
        samples = []
        for scene_name in self.scene_names:
            scene_path = os.path.join(self.raw_data_path, 'sensor_data', scene_name)
            if os.path.exists(scene_path):
                timestamps = [f.split('.')[0] + '.' + f.split('.')[1] for f in os.listdir(scene_path) if f.endswith('.bin')]
            timestamps.sort()
            total_frames = len(timestamps)
            for idx in range(self.history_frames, total_frames - self.future_frames, self.frame_skip):
                current_frame = timestamps[idx]
                past_frames = timestamps[max(0, idx - self.history_frames + 1):idx+1]
                future_frames = timestamps[idx+1:idx + 1 + self.future_frames]
                sample = f"{scene_name} {current_frame} {' '.join(map(str, past_frames))} {' '.join(map(str, future_frames))}"
                samples.append(sample)
        return samples

    def _generate_samples_json(self) -> Dict:
        samples={}

        for scene_name in self.scene_names:
            pkl_file_path = os.path.join(self.root_path, 'groundtruth', self.sensor_type, 'Det_data_infos', f"{scene_name}.pkl")
            if os.path.exists(pkl_file_path):
                scene_name = self.get_samples_in_scenes(scene_name)
            else:
                print(f"scene_path:{pkl_file_path} is not exist")
                continue
            # scene_path = os.path.join(self.raw_data_path, 'sensor_data', scene_name)

            timestamps = list()
            for entry in self.data:
                timestamps.append(entry['point_cloud']['lidar_idx'])
            # timestamps = [f.split('.')[0] + '.' + f.split('.')[1] for f in os.listdir(scene_path) if f.endswith('.bin')]
            timestamps.sort()
            total_frames = len(timestamps)
            scene_samples = {}
            for idx in range(self.history_frames, total_frames - self.future_frames, self.frame_skip):
                current_frame = timestamps[idx]
                past_frames = timestamps[max(0, idx - self.history_frames + 1):idx + 1]
                future_frames = timestamps[idx + 1:idx + self.future_frames + 1]
                scene_samples[str(current_frame)] = {
                    'current_frame': current_frame,
                    'past_frames': past_frames,
                    'future_frames': future_frames
                }
            samples[scene_name] = scene_samples
        return samples

    def save_samples_to_file(self, output_file_path: str):
        with open(output_file_path, 'w') as f:
            for sample in self.samples:
                f.write(sample + '\n')
        print(f"Samples saved to {output_file_path}")

    def save_samples_to_jsonfile(self, output_file_path:str):
        with open(output_file_path, 'w') as f:
            json.dump(self.samples, f, indent=4)
        print(f"JSON Samples saved to {output_file_path}")

    def get_all_scenes(self):
        return self.scenes_names

    def get_samples_in_scenes(self, scene_name):
        pkl_file_path = os.path.join(self.root_path, 'groundtruth', self.sensor_type, 'Det_data_infos', f"{scene_name}.pkl")

        with open(pkl_file_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"The {pkl_file_path} contains {len(self.data)} elements")
        return scene_name

    def get_pointcloud_path(self, scene_name: str, timestamp: float) -> str:
        # 根据场景名和时间戳获取点云数据的路径或数据
        for entry in self.data:
            if entry['point_cloud']['sequence_id'] == scene_name and entry['point_cloud']['lidar_idx'] == str(timestamp):

                lidar_path = f"{self.root_path}/raw_data/{self.sensor_type}/sensor_data/{scene_name}/{timestamp}.bin"
                return lidar_path
        print(f"No pointcloud data for {scene_name} at {timestamp}")
        return None

    def get_pointcloudV_path(self, scene_name: str, timestamp: float) -> str:
        """ get point cloud velocity data path"""
        for entry in self.data:
            if entry['point_cloud']['sequence_id'] == scene_name and entry['point_cloud']['lidar_idx'] == str(timestamp):

                velocity_path = f"{self.root_path}/raw_data/{self.sensor_type}/velocity/{scene_name}/{timestamp}.bin"

                if not os.path.exists(velocity_path):
                    velocity_path = f"{self.root_path}/raw_data/{self.sensor_type}/rel_velocity/{scene_name}/{timestamp}.bin"
                return velocity_path
        print(f"No pointcloud velocity data for {scene_name} at {timestamp}")
        return None

    def get_pose(self, scene_name: str, timestamp: float) -> Dict:
        # 根据场景名和时间戳获取位姿信息
        for entry in self.data:
            if entry['point_cloud']['sequence_id'] == scene_name and entry['point_cloud']['lidar_idx'] == str(timestamp):
                pose_path = f"{self.root_path}/raw_data/{self.sensor_type}/pose/{scene_name}/{timestamp}.txt"
                return pose_path
        return None

    def read_pose_data(self, pose_path:str) -> Dict:
        with open(pose_path, 'r') as f:

            # [1:] fliter the key
            pose_data = f.readline().strip().split()[1:]
            pose_data = list(map(float, pose_data))
            assert len(pose_data) == 16, "Pose data must contain 16 elements"
            transformation_matrix = np.array(pose_data).reshape(4, 4)

            translation = transformation_matrix[:3, 3]
            rotation = transformation_matrix[:3, :3]
            rotation = rotation_matrix_to_quat(rotation)
            pose_info = {
                'translation': translation,
                'rotation': rotation
            }
            return pose_info
        return None

    def read_pose_data_v2(self, pose_path, return_quat=True):
        raw_pose = dict()
        with open(pose_path, "r") as f:
            seqs = f.readlines()
            for seq in seqs:
                cur_seq_list = seq.split(":", 1)
                key_, val_ = cur_seq_list[0], cur_seq_list[1]
                val_ = np.fromstring(val_, sep=" ")
                raw_pose[key_] = val_

        pose = raw_pose["Vehicle_Pose"].reshape(4, 4)
        if not return_quat:
            return pose
        else:
            pose_info = {
                'translation': pose[:3, 3],
                'rotation': rotation_matrix_to_quat(pose[:3, :3])
            }
            return pose_info

    def get_calibrated_sensor(self, scene_name: str, timestamp: float) -> Dict:
        # 根据场景名和时间戳获取校准传感器的信息
        for entry in self.data:
            if entry['point_cloud']['sequence_id'] == scene_name and entry['point_cloud']['lidar_idx'] == timestamp:
                pose_path = f"{self.root_path}/raw_data/{self.sensor_type}/calib/{scene_name}/{timestamp}.txt"
                return pose_path
        return None

    def _map_class(self, original_class:str) -> str:
        """Map the original class to our class"""

        return self.cls_map.get(original_class, "other")

    def get_instance_box(self, scene_name: str, timestamp: float, instance_name: str) -> Tuple[Dict, str, str]:
        # 根据场景名、时间戳和实例名获取实例的边界框
        for entry in self.data:
            if entry['point_cloud']['sequence_id'] == scene_name and entry['point_cloud']['lidar_idx'] == str(timestamp):
                for i, name in enumerate(entry['annos']['ID']):
                    if name == instance_name:
                        box = {
                            'center': entry['annos']['gt_boxes_lidar'][i][:3],
                            'lwh': entry['annos']['gt_boxes_lidar'][i][3:6],
                            'orientation': entry['annos']['gt_boxes_lidar'][i][6:10]
                        }
                        attr = entry['annos']['num_points_in_gt'][i]
                        return box, attr, instance_name
        return None, None, None

    def get_previous_timestamp(self, scene_name: str, timestamp: float) -> float:
        # 获取给定时间戳之前的时间戳
        timestamps = sorted([entry['point_cloud']['lidar_idx'] for entry in self.data if entry['point_cloud']['sequence_id'] == scene_name])
        idx = timestamps.index(timestamp)
        if idx > 0:
            return timestamps[idx - 1]
        return None

    def get_next_timestamp(self, scene_name: str, timestamp: float) -> float:
        # 获取给定时间戳之后的时间戳
        timestamps = sorted([entry['point_cloud']['lidar_idx'] for entry in self.data if entry['point_cloud']['sequence_id'] == scene_name])
        idx = timestamps.index(timestamp)
        if idx < len(timestamps) - 1:
            return timestamps[idx + 1]
        return None

    def get_annotations_for_timestamp(self, scene_name: str, timestamp: float) -> List[Dict]:
        # 获取给定时间戳的注释信息

        gt_path = f"{self.root_path}/groundtruth/{self.sensor_type}/Det_data_infos/{scene_name}.pkl"
        # print("The gt_path:{}".format(gt_path))
        for entry in self.data:
            if entry['point_cloud']['sequence_id'] == scene_name and entry['point_cloud']['lidar_idx'] == str(timestamp):
                annotations = []
                for i in range(len(entry['annos']['ID'])):
                    ann = {
                        'ann_token': entry['annos']['ID'][i],
                        'category_name': self._map_class(entry['annos']['name'][i]),
                        'instance_token': entry['annos']['ID'][i],
                        'box': {
                            'center': entry['annos']['gt_boxes_lidar'][i][:3],
                            'lwh': entry['annos']['gt_boxes_lidar'][i][3:6],
                            'orientation': entry['annos']['gt_boxes_lidar'][i][6:10]
                        }
                    }
                    annotations.append(ann)
                return annotations
        return []

    def get_annotation_by_name(self, scene_name: str, timestamp: float, category_name: str) -> List[Dict]:
        # 根据场景名、时间戳和类别名获取注释信息
        for entry in self.data:
            if entry['point_cloud']['sequence_id'] == scene_name and entry['point_cloud']['lidar_idx'] == str(timestamp):
                annotations = []
                for i in range(len(entry['annos']['name'])):
                    if entry['annos']['name'][i] == category_name:
                        ann = {
                            'ann_token': entry['annos']['ID'][i],
                            'category_name': self._map_class(entry['annos']['name'][i]),
                            'instance_token': entry['annos']['ID'][i],
                            'box': {
                                'center': entry['annos']['gt_boxes_lidar'][i][:3],
                                'lwh': entry['annos']['gt_boxes_lidar'][i][3:6],
                                'orientation': entry['annos']['gt_boxes_lidar'][i][6]
                            }
                        }
                        annotations.append(ann)
                return annotations
        return []

if __name__ == "__main__":
    # Test CustomDataset
    scene_name = "task_10_30-13_38_13_14.db3_seq_7"
    # timestamp = "1698646900.682504"
    timestamp = '1698646900.781587'
    pickle_filepath = f"data/HaData/groundtruth/at128_0/Det_data_infos/{scene_name}.pkl"
    # dataset = CustomDataset(root_path="data/HaData", sensor_type="at128_0", )
    dataset = CustomDataset(root_path="data/HaData", sensor_type='at128_0', data_type="train", \
        frame_skip=3, history_frames=25, future_frames=20)

    # get current frame in one scene info
    curr_scene = dataset.get_samples_in_scenes(scene_name=scene_name)
    lidar_path = dataset.get_pointcloud_path(scene_name=scene_name, timestamp=timestamp)
    pose_path = dataset.get_pose(scene_name=scene_name, timestamp=timestamp)
    prev_t = dataset.get_previous_timestamp(scene_name=scene_name, timestamp=timestamp)
    next_t = dataset.get_next_timestamp(scene_name=scene_name, timestamp=timestamp)
    annotations = dataset.get_annotations_for_timestamp(scene_name=scene_name, timestamp=timestamp)
    print("The lidar path: {}\n The pose path: {}\n The previous timestamp: {}\n The next timestamp: {}\n The annotations: {}".format(lidar_path, pose_path, prev_t, next_t, annotations))
