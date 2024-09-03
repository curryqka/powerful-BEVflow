import numpy as np
from functools import reduce
from abc import ABC,abstractmethod
import sys,cv2
sys.path.insert(0, "/home/qiankangan/MotionNet")
from HeTools.Hedata import CustomDataset
from HeTools.He_utils import transform_matrix, view_points
from typing import List, Dict, Tuple, Any
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from haplugin.utils.box_utils import boxes_to_corners_3d, corners_3d_to_boxes3d
class LidarPointCloud(ABC):
    """
    Abstract class for a point cloud.
    """

    def __init__(self, points:np.array, sensor_type = "at128_0"):
        """
        Initialize a point cloud
        """
        self.sensor_type = sensor_type
        if sensor_type == 'at128_0':
            assert points.shape[0] == self.nbr_dims(), 'Error: Point cloud should have {} dimensions.'.format(self.nbr_dims())
        else:
            assert points.shape[0] == self.nbr_dims() + 1, 'Error: Point cloud should have {} dimensions.'.format(self.nbr_dims())
        self.points = points

    # def __init__(self, points):
    #     self.points = points
    @staticmethod
    @abstractmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        pass


    @classmethod
    @abstractmethod
    def from_file(cls, file_name: str) -> 'LidarPointCloud':
        """
        Loads point cloud from disk.
        :param file_name: Path of the pointcloud file on disk.
        :return: PointCloud instance.
        """
        pass

    @classmethod
    def from_file_multisweep_bf_sample_data(cls, dataset:CustomDataset, scene_name, timestamps, \
                                            nsweeps_back: int = 5,
                                            nsweeps_forward: int = 5,
                                            return_trans_matrix: bool = False,
                                            min_distance=1.0):
        """
        Return a point cloud that aggregates multiple sweeps. The sweeps trace back as well as into the future.
        As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
        As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
        :param dataset: The dataset instance.
        :param scene_name: The name of the scene.
        :param timestamps: List of timestamps (current, past, and future).
        :param return_trans_matrix: Whether need to return the transformation matrix.
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """

        # Initialize
        if dataset.sensor_type == 'at128_0':
            points = np.zeros((cls.nbr_dims(), 0))
        else:
            points = np.zeros((cls.nbr_dims() + 1, 0))
        all_pc = cls(points, sensor_type=dataset.sensor_type)
        all_times = np.zeros((1, 0))
        trans_matrix_list = list()

        # Get reference pose and timestamp
        ref_timestamp = timestamps[nsweeps_back - 1]
        ref_pose_rec_path = dataset.get_pose(scene_name, ref_timestamp)
        ref_pose_rec = dataset.read_pose_data_v2(ref_pose_rec_path, return_quat=False)
        ref_cs_rec = dataset.get_calibrated_sensor(scene_name, ref_timestamp)
        ref_time = ref_timestamp

        # Homogeneous transform from ego car frame to reference frame
        # ref_from_car = transform_matrix(ref_cs_rec['translation'], ref_cs_rec['rotation'], inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        # ref_from_global = transform_matrix(ref_pose_rec['translation'], ref_pose_rec['rotation'], inverse=True)
        ref_from_global = np.linalg.inv(ref_pose_rec)

        # Aggregate current and past sweeps
        for k in range(nsweeps_back):
            timestamp = timestamps[nsweeps_back - 1 - k]

            if dataset.sensor_type == 'wt128_0':
                current_pc = cls.from_file(dataset.get_pointcloud_path(scene_name, timestamp), \
                    dataset.get_pointcloudV_path(scene_name, timestamp), dataset=dataset)
            else:
                current_pc = cls.from_file(dataset.get_pointcloud_path(scene_name, timestamp), \
                    dataset=dataset)

            current_pose_rec_path = dataset.get_pose(scene_name, timestamp)
            current_pose_rec = dataset.read_pose_data_v2(current_pose_rec_path, return_quat=False)

            # global_from_current = transform_matrix(current_pose_rec['translation'], current_pose_rec['rotation'], inverse=False)
            global_from_current = current_pose_rec

            # current_cs_rec = dataset.get_calibrated_sensor(scene_name, timestamp)
            # car_from_current = transform_matrix(current_cs_rec['translation'], current_cs_rec['rotation'], inverse=False)

            # Fuse four transformation matrices into one and perform transform
            trans_matrix = reduce(np.dot, [ref_from_global, global_from_current])
            current_pc.transform(trans_matrix)

            # Collect the transformation matrix
            trans_matrix_list.append(trans_matrix)

            # Remove close points and add timevector
            current_pc.remove_close(min_distance)
            time_lag = float(ref_time) - float(timestamp)  # positive difference
            times = time_lag * np.ones((1, current_pc.nbr_points()))
            all_times = np.hstack((all_times, times))

            # Merge with key pc
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

        # Aggregate the future sweeps
        for k in range(nsweeps_forward):
            timestamp = timestamps[nsweeps_back + k]

            if dataset.sensor_type == 'wt128_0':
                current_pc = cls.from_file(dataset.get_pointcloud_path(scene_name, timestamp), \
                    dataset.get_pointcloudV_path(scene_name, timestamp), dataset)
            else:
                current_pc = cls.from_file(dataset.get_pointcloud_path(scene_name, timestamp), \
                    dataset=dataset)

            current_pose_rec_path = dataset.get_pose(scene_name, timestamp)
            current_pose_rec = dataset.read_pose_data_v2(current_pose_rec_path, return_quat=False)

            # global_from_current = transform_matrix(current_pose_rec['translation'], current_pose_rec['rotation'], inverse=False)
            global_from_current = current_pose_rec
            # current_cs_rec = dataset.get_calibrated_sensor(scene_name, timestamp)
            # car_from_current = transform_matrix(current_cs_rec['translation'], current_cs_rec['rotation'], inverse=False)

            # Fuse four transformation matrices into one and perform transform
            # trans_matrix = reduce(np.dot, [ref_from_global, global_from_current])
            trans_matrix = np.dot(ref_from_global, global_from_current)
            current_pc.transform(trans_matrix)

            # Remove close points and add timevector
            # current_pc.remove_close(min_distance)
            time_lag = float(ref_time) - float(timestamp)  # negative difference
            times = time_lag * np.ones((1, current_pc.nbr_points()))
            all_times = np.hstack((all_times, times))

            # Merge with key pc
            all_pc.points = np.hstack((all_pc.points, current_pc.points))

        trans_matrix_list = np.stack(trans_matrix_list, axis=0)

        if return_trans_matrix:
            return all_pc, np.squeeze(all_times, 0), trans_matrix_list
        else:
            return all_pc, np.squeeze(all_times, 0)

    @classmethod
    def get_instance_boxes_multisweep_sample_data(cls,
                                                  dataset: CustomDataset,
                                                  scene_name: str,
                                                  ref_timestamp: float,
                                                  instance_token: str,
                                                  nsweeps_back: int = 5,
                                                  nsweeps_forward: int = 5) -> \
            Tuple[List[Dict], np.array, List[str], List[str]]:
        """
        Return the bounding boxes associated with the given instance. The bounding boxes are across different sweeps.
        For each bounding box, we need to map its (global) coordinates to the reference frame.
        :param dataset: The dataset instance.
        :param scene_name: The name of the scene.
        :param ref_timestamp: The reference timestamp.
        :param instance_token: The current selected instance.
        :param nsweeps_back: Number of sweeps to aggregate. The sweeps trace back.
        :param nsweeps_forward: Number of sweeps to aggregate. The sweeps are obtained from the future.
        :return: (list of bounding boxes, the time stamps of bounding boxes, attribute list, category list)
        """

        # Init
        box_list = list()
        all_times = list()
        attr_list = list()  # attribute list
        cat_list = list()  # category list

        # Get reference pose and timestamp
        ref_pose_rec_path = dataset.get_pose(scene_name, ref_timestamp)
        ref_pose_rec = dataset.read_pose_data_v2(ref_pose_rec_path, return_quat=False)
        # Homogeneous transformation matrix from global to _current_ ego car frame
        # ref_from_global = transform_matrix(ref_pose_rec['translation'], ref_pose_rec['rotation'], inverse=True)
        ref_from_global = np.linalg.inv(ref_pose_rec)

        # ref_cs_rec = dataset.get_calibrated_sensor(scene_name, ref_timestamp)
        ref_time = ref_timestamp

        # Get the bounding boxes across different sweeps
        boxes = list()

        # Move backward to get the past annotations
        current_timestamp = ref_timestamp
        for _ in range(nsweeps_back):
            box, attr, cat = dataset.get_instance_box(scene_name, current_timestamp, instance_token)

            # qka 20240813

            boxes.append(box)  # It is possible the returned box is None
            attr_list.append(attr)
            cat_list.append(cat)

            time_lag = float(ref_time) - float(current_timestamp)  # positive difference
            all_times.append(time_lag)

            previous_timestamp = dataset.get_previous_timestamp(scene_name, current_timestamp)
            if previous_timestamp is None:
                break
            else:
                current_timestamp = previous_timestamp

        # Move forward to get the future annotations
        current_timestamp = ref_timestamp

        next_timestamp = dataset.get_next_timestamp(scene_name, current_timestamp)
        if next_timestamp is not None:
            current_timestamp = next_timestamp

            for _ in range(nsweeps_forward):
                box, attr, cat = dataset.get_instance_box(scene_name, current_timestamp, instance_token)
                if box is not None:
                 # qka 20240813

                    #  TODO: rewrite box info: dict -> numpy
                    box_numpy = np.array([box['center'][0], box['center'][1], box['center'][2], \
                                            box['lwh'][0], box['lwh'][1], box['lwh'][2],
                                            box['orientation'][0]]).reshape(1, 7)
                    corner_rotate =  boxes_to_corners_3d(box_numpy)

                    current_pose_rec_path = dataset.get_pose(scene_name, current_timestamp)
                    current_pose_rec = dataset.read_pose_data_v2(current_pose_rec_path, return_quat=False)
                    # global_from_current = transform_matrix(current_pose_rec['translation'], current_pose_rec['rotation'], inverse=False)
                    global_from_current = current_pose_rec

                    # current_cs_rec = dataset.get_calibrated_sensor(scene_name, timestamp)
                    # car_from_current = transform_matrix(current_cs_rec['translation'], current_cs_rec['rotation'], inverse=False)

                    # Fuse four transformation matrices into one and perform transform
                    # trans_matrix = np.dot(ref_from_global, global_from_current)

                    # corner_rotate = np.matmul(trans_matrix[:3, :3], corner_rotate.transpose(0, 2, 1))
                    # corner_rotate = trans_matrix[:3, 3].reshape(3,1) + corner_rotate

                    # import numpy as np

                    # 假设 trans_matrix 是 4x4 的转换矩阵
                    trans_matrix = np.dot(ref_from_global, global_from_current)

                    # corner_rotate 是形状为 (1, 8, 3) 的输入数据
                    # 扩展到齐次坐标（添加最后一列 1）
                    ones = np.ones((corner_rotate.shape[0], corner_rotate.shape[1], 1))
                    corner_rotate_homogeneous = np.concatenate((corner_rotate, ones), axis=2)

                    # 转换矩阵与扩展后的数据进行矩阵乘法
                    transformed_homogeneous = np.matmul(corner_rotate_homogeneous, trans_matrix.T)

                    # 只取前三列作为最终结果
                    corner_rotate_transformed = transformed_homogeneous[:, :, :3]

                    # print(corner_rotate.shape)
                    new_box = corners_3d_to_boxes3d(corner_rotate_transformed)
                    # print(new_box.shape)
                    new_box_dict = dict(
                        center=new_box[0][0:3],
                        lwh=new_box[0][3:6],
                        orientation=new_box[0][-1:]
                    )
                    boxes.append(new_box_dict)  # It is possible the returned box is None
                else:
                    boxes.append(box)
                attr_list.append(attr)

                cat_list.append(cat)

                time_lag = float(ref_time) - float(current_timestamp)  # negative difference
                all_times.append(time_lag)

                next_timestamp = dataset.get_next_timestamp(scene_name, current_timestamp)
                if next_timestamp is None:
                    break
                else:
                    current_timestamp = next_timestamp

        # Map the bounding boxes to the local sensor coordinate
        for box in boxes:
            # if box is not None:
            #     # Move box to ego vehicle coord system
            #     rotation_obj = R.from_quat(ref_pose_rec['rotation'].inverse.elements)

            #     # Move box to global coord system
            #     box['center'] -= ref_pose_rec['translation']
            #     box['center'] = rotation_obj.apply(box['center'])


            #     euler_angles =np.array([0,0,box['orientation'][0]])

            #     orientation_rotation = R.from_euler('xyz', euler_angles)
            #     new_orientation = rotation_obj * orientation_rotation
            #     box['orientation'] = np.array([new_orientation.as_euler('xyz')[2]])
                # box['orientation'] = Quaternion(ref_pose_rec['rotation']).inverse.rotate(box['orientation'])

                # box.translate(-np.array(ref_pose_rec['translation']))
                # box.rotate(Quaternion(ref_pose_rec['rotation']).inverse)

                # Move box to sensor coord system
                # box['center'] -= np.array(ref_cs_rec['translation'])
                # box['orientation'] = Quaternion(ref_cs_rec['rotation']).inverse.rotate(box['orientation'])

            box_list.append(box)

        return box_list, np.array(all_times), attr_list, cat_list


    def nbr_points(self) -> int:
        """
        Returns the number of points.
        :return: Number of points.
        """
        return self.points.shape[1]

    def subsample(self, ratio: float) -> None:
        """
        Sub-samples the pointcloud.
        :param ratio: Fraction to keep.
        """
        selected_ind = np.random.choice(np.arange(0, self.nbr_points()), size=int(self.nbr_points() * ratio))
        self.points = self.points[:, selected_ind]

    def remove_close(self, radius: float) -> None:
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """

        x_filt = np.abs(self.points[0, :]) < radius
        y_filt = np.abs(self.points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        self.points = self.points[:, not_close]

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix: np.ndarray) -> None:
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix: np.ndarray) -> None:
        """
        Applies a homogeneous transform.
        :param transf_matrix: <np.float: 4, 4>. Homogenous transformation matrix.
        """
        self.points[:3, :] = transf_matrix.dot(np.vstack((self.points[:3, :], np.ones(self.nbr_points()))))[:3, :]


class HeLidarPointCloud(LidarPointCloud):
    """
    A class to store a point cloud in HeLiNK format.
    """
    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """

        return 4

    @classmethod
    def from_file(cls, file_name: str, vel_file_name=None, dataset:CustomDataset = None) -> 'HeLidarPointCloud':
        """
        Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
        :param file_name: Path of the pointcloud file on disk.
        :return: LidarPointCloud instance (x, y, z, intensity).
        """

        assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)

        if dataset.sensor_type == "at128_0":
            scan = np.fromfile(file_name, dtype=np.float32)
            points = scan.reshape((-1, 4))[:, :cls.nbr_dims()]
            # points = scan.reshape((128, 1200, -1))[:, :, :cls.nbr_dims()]
            return cls(points.T)
        elif dataset.sensor_type == "wt128_0":
            scan = np.fromfile(file_name, dtype=np.float32)
            points = scan.reshape((-1, 4))[:, :cls.nbr_dims()]
            # points = scan.reshape((128, 1200, -1))[:, :, :cls.nbr_dims()]
            vel_scan = np.fromfile(vel_file_name, dtype=np.float32)
            vel = vel_scan.reshape((-1, 1))[:, :1]

            points = np.concatenate((points, vel), axis=1)

            return cls(points.T, sensor_type="wt128_0")

class Box:
    """
    Simple data class representing a 3D box as (x, y, z, l, w, h, ry) or w(x, y, z, l, w, h, label, score, v)
    """

    def __init__(self,
                 center: List[float],\
                 size: List[float],
                 orientation: float,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 id: str = None) -> None:
        """
        Params:
        center: x, y, z
        size: l, w, h
        yaw: ry

        # Box center
        self.x = center[0]
        self.y = center[1]
        self.z = center[2]

        # Box size
        self.l = size[0]
        self.w = size[1]
        self.h = size[2]

        # Box orientation
        self.ry = yaw

        # Additional attributes
        self.label = label
        self.score = score
        self.velocity = velocity
        self.name = name
        self.id = id

        """

        assert len(center) == 3, "Box center should be a list of length 3"
        assert len(size) == 3, "Box size should be a list of length 3"

        self.center = np.array(center)
        self.lwh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity) if not np.isnan(velocity[0]) else velocity
        self.name = name
        self.token = id

    def __eq__(self, other) -> bool:
        """Check if two box instances are equal
        params:
        other: other Box instance
        return: bool
        """
        center = np.allclose(self.center, other.center)
        lwh = np.allclose(self.lwh, other.lwh)
        yaw = np.isclose(self.orientation, other.yaw)
        label = (self.label == other.label) or (np.islan(self.label) and np.isnan(other.label))
        score = (self.score == other.score) or (np.isnan(self.score) and np.isnan(other.score))
        vel = np.allclose(self.velocity, other.velocity) or (np.isnan(self.velocity[0]) and np.isnan(other.velocity))
        return center and lwh and yaw and label and score and vel

    def __repr__(self) -> str:
        """Return a string representation of the box instances
        return: str
        """
        return "Box(center={}, lwh={}, yaw={}, label={}, score={}, velocity={}, name={}, token={})". \
             format(self.center, self.lwh, self.orientation, self.label, self.score, self.velocity, self.name, self.token)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Compute the rotation matrix for the yaw

        return: 3x3 rotation matrix
        """
        cos_yaw = np.cos(self.orientation)[0]
        sin_yaw = np.sin(self.orientation)[0]

        return np.array([[cos_yaw, -sin_yaw, 0],
                         [sin_yaw, cos_yaw, 0],
                         [0, 0, 1]])

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation to the box.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        self.center = self.center + x

    def rotate(self, yaw:float) -> None:
        """
        Applies a rotation.
        :param : yaw angle
        Rotation matrix.
        """
        rot_matrix = self.rotation_matrix
        self.center[:2] = np.dot(rot_matrix[:2, :2], self.center[:2])
        self.orientation = self.orientation + yaw
        self.velocity = np.dot(rot_matrix[:2, :2], self.velocity[:2])

    def corners(self, lwh_factor: float = 1.0) -> np.ndarray:
        """
        Compute the 3D corners of the box.
        params:
            lwh_factorL Scale factor for the box dimensions
            return 3x8 array of corner points. First Four corners face forward, last four face backward.
        """
        l, w, h = self.lwh * lwh_factor

        # 3D corner points. relative to C, Z points up, Y to the left, X forward
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))


        # rotate
        corners[:2, :] = np.dot(self.rotation_matrix[:2, :2], corners[:2, :])

        # Translate
        x, y, z = self.center
        corners[0, :] += x
        corners[1, :] += y
        corners[2, :] += z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Compute the 4 bottom corners of the Box.
        reverse clock-wise
        :return: 3x4 array of bottom corner points.
        """

        return self.corners()[:, [2, 3, 7, 6]]

    def render(self,
               axis,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 1.2):
        """
        Render the Box in a Matplotlib axis.

        :param axis: Matplotlib axis to draw the box on.
        :param view: 3x3 projection matrix.
        :param normalize: Whether to normalize the coordinates.
        :param colors: Tuple of colors for front, rear, and side edges.
        :param linewidth: Line width for drawing.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth)

        # Draw front and rear rectangles
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth)

    def render_vis(self,
                   cat: str,
                   axis,
                   view: np.ndarray = np.eye(3),
                   normalize: bool = False,
                   colors: Tuple = ('b', 'r', 'k'),
                   linewidth: float = 1.2):
        """
        Render the Box with category-specific colors in a Matplotlib axis.

        :param cat: Category of the box, used to determine color.
        :param axis: Matplotlib axis to draw the box on.
        :param view: 3x3 projection matrix.
        :param normalize: Whether to normalize the coordinates.
        :param colors: Tuple of colors for front, rear, and side edges.
        :param linewidth: Line width for drawing.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        # determined color based on category
        if cat.startswith('vehicle'):
            color = "royalblue"
        elif cat == "Pedestrain":
            color = 'c'
        elif cat == "Cyclist":
            color = 'r'
        else:
            color = 'darkorange'

        # Skip rendering for movable objects
        if cat.startswith('movable'):
            return

        # Draw the sides
        for i in range(2):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=color, linewidth=linewidth)

        # Draw front and rear lines
        axis.plot([corners.T[0][0], corners.T[1][0]],
                  [corners.T[0][1], corners.T[1][1]],
                  color=color, linewidth=linewidth)

        axis.plot([corners.T[4][0], corners.T[5][0]],
                  [corners.T[4][1], corners.T[5][1]],
                  color=color, linewidth=linewidth)

    def render_cv2(self,
                   im: np.ndarray,
                   view: np.ndarray = np.eye(3),
                   normalize: bool = False,
                   colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
                   linewidth: int = 1.2) -> None:
        """
        Render the Box using OpenCV.

        :param im: Image array (width, height, 3) in BGR order.
        :param view: 3x3 projection matrix.
        :param normalize: Whether to normalize the coordinates.
        :param colors: Tuple of BGR colors for front, rear, and side edges.
        :param linewidth: Line width for drawing.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                cv2.line(im,
                         (int(prev[0]), int(prev[1])),
                         (int(corner[0]), int(corner[1])),
                         color, linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            cv2.line(im,
                     (int(corners.T[i][0]), int(corners.T[i][1])),
                     (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                     colors[2][::-1], linewidth)

        # Draw front and rear rectangles
        draw_rect(corners.T[:4], colors[0][::-1])
        draw_rect(corners.T[4:], colors[1][::-1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[0:2], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        cv2.line(im,
                 (int(center_bottom[0]), int(center_bottom[1])),
                 (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                 colors[0][::-1], linewidth)


"""
Add new code to process the FMCW lidar data.
"""
def calculate_voxel_speed(padded_voxel_points, unique_indices, num_points_in_voxel, \
                            points, num_T, seed):
    """
    Calculate the speed of each voxel in the padded voxel points.
    params:
        padded_voxel_points: padded voxel points
        unique_indices: unique indices of the voxel points
        num_points_in_voxel: number of points in each voxel
        points: original points
        num_T: number of max points in each voxel
    """
    if seed is not None:
        np.random.seed(seed)
    voxel_speeds = np.zeros((unique_indices.shape[0], 1), dtype=np.float32)
    for i, v in enumerate(zip(unique_indices, num_points_in_voxel)):
        if v[1] < num_T:
            voxel_speeds = np.mean(points[v[0]:v[0]+v[1], 4], axis = 0)
        else:
            inds = np.random.choice(v[1], num_T)
            voxel_speeds[i, :] = np.mean(points[v[0]+inds, 4], axis = 0)
    return voxel_speeds

def calculate_voxel_direction(padded_voxel_points, unique_indices, num_points_in_voxel, \
                            points, num_T, seed):
    """
    Calculate the direction of each voxel in the padded voxel points.
    """
    if seed is not None:
        np.random.seed(seed)
    # save the direction of each voxel
    voxel_directions = np.zeros((unique_indices.shape[0], 3), dtype=np.float32)

    for i, v in enumerate(zip(unique_indices, num_points_in_voxel)):
        if v[1] < num_T:
            direction_vectors = points[v[0]:v[0] + v[1], :3]
        else:
            inds = np.random.choice(v[1], num_T)
            direction_vectors = points[v[0] + inds, :3]

        # calculate the direction of each voxel [0,0,0]
        directions = direction_vectors
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        norm_directions = directions / np.where(norm == 0, 1, norm)

        voxel_directions[i, :] = np.mean(norm_directions, axis=0)
    return voxel_directions
