
from enum import IntEnum
from typing import Tuple
import os
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
def rotation_matrix_to_quat(matrix):
    rotation = R.from_matrix(matrix=matrix).as_quat()
    
    return Quaternion(rotation)


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

def transform_corners(corners:np.ndarray, trans_matrix):
    # corner_rotate 是形状为 (1, 8, 3) 的输入数据
    # 扩展到齐次坐标（添加最后一列 1）
    ones = np.ones((corners.shape[0], corners.shape[1], 1))
    corner_rotate_homogeneous = np.concatenate((corners, ones), axis=2)

    # 转换矩阵与扩展后的数据进行矩阵乘法
    transformed_homogeneous = np.matmul(corner_rotate_homogeneous, trans_matrix.T)

    # 只取前三列作为最终结果
    corner_rotate_transformed = transformed_homogeneous[:, :, :3]
    return corner_rotate_transformed

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm

def yaw_to_rotation_matrix(yaw):
    """
    Convert yaw angle to 3x3 rotation matrix.
    :param yaw: Yaw angle in radians.
    :return: 3x3 rotation matrix.
    """
    cos_yaw = np.cos(yaw)[0]
    sin_yaw = np.sin(yaw)[0]
    return np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])

def check_pc(pc_file_path, point_cloud_range = [0, -32, -3, 64, 32, 2]):
    frames = os.listdir(pc_file_path)
    frames = [os.path.join(pc_file_path, frame) for frame in frames if frame.endswith(".bin")]
    for idx, frame in enumerate(frames):
    # print(frames)
    # save_data_dict
    # 1708911118.192297
    # 1708911112.392067
    # if frame.split('/')[-1] == "1708911112.392067.bin":
        
        if idx >= 0 and idx <= 1e3:
            points = np.fromfile(frame, dtype=np.float32).reshape(-1, 5)
            mask = (
                (points[:, 0] >= point_cloud_range[0])
                & (points[:, 1] >= point_cloud_range[1])
                & (points[:, 2] >= point_cloud_range[2])
                & (points[:, 0] <= point_cloud_range[3])
                & (points[:, 1] <= point_cloud_range[4])
                & (points[:, 2] <= point_cloud_range[5])
            )
            # breakpoint()
            points = points[mask]
            if points.shape[0] < 5e4:
                print(f"The points frame: {frame} less than 50000!")
                return False
                # print(points)
            
            # print(frame)
            # print(idx, points.shape)
    return True