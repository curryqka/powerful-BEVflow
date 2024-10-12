
from enum import IntEnum
from typing import Tuple
import os
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import matplotlib
import matplotlib.pyplot as plt

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

def check_pc(pc_file_path, point_cloud_range = [0, -32, -3, 64, 32, 2], show=False, \
            img_save_dir="/home/qiankangan"):
    frames = os.listdir(pc_file_path)
    frames = [os.path.join(pc_file_path, frame) for frame in frames if frame.endswith(".bin")]
    for idx, frame in enumerate(frames):
    # print(frames)
    # save_data_dict
    # 1708911118.192297
    # 1708911112.392067
    # if frame.split('/')[-1] == "1708911112.392067.bin":
        
        if idx >= 0 and idx <= 1e3:
            # breakpoint()
            scan = np.fromfile(frame, dtype=np.float32)

            
            
            # print("scene:{},shape:{}".format(pc_file_path, scan.shape))
            if show:
                if scan.shape[0]==64000*6:
                    points = scan.reshape(-1, 6).T
                else:
                    points = scan.reshape(-1, 5).T
                fig, ax = plt.subplots(1, 1, figsize=(20, 8))
                x_lim = [0, 64]
                y_lim = [-32, 32]   
                v_mask = np.abs(points[-2, :]) < 1
                v_mask_ = np.abs(points[-2, :]) >= 1

                # breakpoint()

                # ax[2].scatter(points[0, :], points[1, :], c=points[-1, :], s=1)
                ax.scatter(points[0, v_mask], points[1, v_mask], c='blue', s=1)
                ax.scatter(points[0, v_mask_], points[1, v_mask_], c='red', s=1)
                ax.set_xlim(x_lim[0], x_lim[1])
                ax.set_ylim(y_lim[0], y_lim[1])
                ax.axis("off")
                ax.set_aspect("equal")
                # TODO: quiver
                # U = 
                # qk = ax[1].quiver(idx_x, idx_y, U, V, angles="xy", scale_units="xy", scale=1, color=color_map[k])
                # ax[1].quiverkey(qk[k], X=0.0 + k / 5.0, Y=1.1, U=20, label=cat_names[k], labelpos="E")
                ax.title.set_text("LIDAR data with v")
                plt.savefig(os.path.join(img_save_dir, "frame.png"))
                ax.clear()
            assert scan.shape[0]==128*500*5
            points = scan.reshape(-1,5)
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
            if points.shape[0] < 4e4 and np.where(points[:, 0]!=0)[0].shape[0] < points.shape[0]/2:
                # breakpoint()
                print(f"The points frame: {frame} less than 40000!")
                return False
                # print(points)
            
            # print(frame)
            # print(idx, points.shape)
    return True