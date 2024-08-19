import numpy as np
import matplotlib.pyplot as plt

def visualize_voxel_frames(voxel_frames, x_lim=(0, 256), y_lim=(0, 256)):
    """
    可视化voxel占据栅格的每一帧。
    
    参数:
    voxel_frames (numpy.ndarray): 输入的voxel占据栅格帧，形状为(5, 256, 256, 13)。
    x_lim (tuple): x轴的限制范围。
    y_lim (tuple): y轴的限制范围。
    """
    num_frames, width, height, depth = voxel_frames.shape

    fig, ax = plt.subplots(1, num_frames, figsize=(15, 5))
    
    for frame_idx in range(num_frames):
        frame = voxel_frames[frame_idx]
        
        # 提取占据体素的坐标
        occupied_points = np.argwhere(frame > 0)
        print(occupied_points)
        # 提取x, y, z坐标
        x_coords = occupied_points[:, 0]
        y_coords = occupied_points[:, 1]
        z_coords = occupied_points[:, 2]

        # 绘制散点图
        ax.scatter(x_coords, y_coords, c='black', s=0.005)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.axis('off')
        ax.set_aspect('equal')
        ax.title.set_text(f'Frame {frame_idx + 1}')

    plt.savefig('output.png')

# 示例使用
if __name__ == "__main__":
    voxel_frames = np.random.randint(0, 2, (1, 256, 256, 13))  # 随机生成5帧voxel占据栅格
    print(voxel_frames==0)
    visualize_voxel_frames(voxel_frames)