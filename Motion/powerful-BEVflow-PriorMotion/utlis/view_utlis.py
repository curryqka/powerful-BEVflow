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

def plot_true_vs_measured(future_disp_mean, cur_disp_mean, save_path=None, show=False):
    """
    Plot true vs measured values to highlight errors.

    Parameters:
    - future_disp_mean: List or array of true future displacements.
    - cur_disp_mean: List or array of measured future displacements.
    - save_path: Path to save the generated plot.
    """
    # Convert lists to numpy arrays for easier manipulation
    future_disp_mean = np.array(future_disp_mean)
    cur_disp_mean = np.array(cur_disp_mean)

    # Plotting
    plt.figure(figsize=(10, 10))
    plt.scatter(future_disp_mean, cur_disp_mean, label='Measurements')
    plt.plot([future_disp_mean.min(), future_disp_mean.max()],
             [future_disp_mean.min(), future_disp_mean.max()], 'r--', label='Ideal Line')
    plt.xlabel('True Future Displacement')
    plt.ylabel('Measured Future Displacement')
    plt.title('True vs Measured Future Displacement')
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

def plot_error_distribution(error, save_path=None, show=False):
    """
    Plot the distribution of errors.

    Parameters:
    - error: List or array of errors.
    - save_path: Path to save the generated plot.
    """
    # Convert lists to numpy arrays for easier manipulation
    error = np.array(error)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(error, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
    plt.xlabel('Error(m/s)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Errors')
    plt.grid(axis='y', alpha=0.75)
    plt.xlim(0,15)
    
    # 设置细粒度的刻度

    x_ticks = np.arange(0, 15, 1)
    plt.xticks(x_ticks, fontsize=10)  # 设置 x 轴刻度字体大小
    # y_ticks = np.arange(0, 35, 1)
    # plt.yticks(y_ticks, fontsize=7)  # 设置 y 轴刻度字体大小

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
def plot_error_point(error_vector, save_path=None, show=False):
    """
    Plot the distribution of errors.
    Parameters:
    - error: List or array of errors.
    - save_path: Path to save the generated plot.
    """
    # Convert lists to numpy arrays for easier manipulation
    error_vector = np.array(error_vector)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(error_vector, 'bo', markersize=2, label='Error Points')

    plt.xlabel('Frame Index')
    plt.ylabel('Error(m/s)')
    plt.title('Error Points')

    # 设置细粒度的刻度

    x_ticks = np.arange(0, len(error_vector), 1000)
    plt.xticks(x_ticks, fontsize=7)  # 设置 x 轴刻度字体大小
    y_ticks = np.arange(0, 35, 1)
    plt.yticks(y_ticks, fontsize=7)  # 设置 y 轴刻度字体大小

    plt.legend()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')

    if show:
        plt.show()



def plot_instance_counts_distribution(instance_counts_list, save_path=None, show=False):
    """
    Plot the distribution of instance counts per frame.

    Parameters:
    - instance_counts_list: List of integers representing the number of instances per frame.
    - save_path: Path to save the generated plot.
    """
    # Convert list to numpy array for easier manipulation
    instance_counts = np.array(instance_counts_list)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(instance_counts, bins='auto', color='green', alpha=0.7, rwidth=0.85)
    plt.xlabel('Number of Instances per Frame')
    plt.ylabel('Frequency')
    plt.title('Distribution of Instance Counts per Frame')
    plt.grid(axis='y', alpha=0.75)
    
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

def plot_instance_grids_distribution(instance_grid_counts_list, save_path=None, show=False):
    """
    Plot the distribution of grids per instance across all frames.

    Parameters:
    - instance_grid_counts_list: List where each element is an array representing the grid count per instance in a frame.
    - save_path: Path to save the generated plot.
    """
    # Flatten the list of lists to get a single list of all grid counts
    flat_grid_counts = [grid_count for sublist in instance_grid_counts_list for grid_count in sublist]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(flat_grid_counts, bins='auto', color='orange', alpha=0.7, rwidth=0.85)
    plt.xlabel('Grid Count per Instance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Grids per Instance Across All Frames')
    plt.grid(axis='y', alpha=0.75)
    
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

def test_plot():
    # Load your data dictionary
    ret_dict = np.load('your_data.npy', allow_pickle=True).item()  # Replace with actual path to your .npy file
    
    # Extract necessary data
    future_disp_mean = ret_dict['future_disp_mean']
    cur_disp_mean = ret_dict['cur_disp_mean']
    error_map_mean = ret_dict['error_map_mean']  # Assuming this is the mean error map
    rel_error_map_mean = ret_dict['rel_error_map_mean']  # Assuming this is the mean relative error map
    
    # Calculate errors
    error = future_disp_mean - cur_disp_mean
    
    # Generate plots
    plot_true_vs_measured(future_disp_mean, cur_disp_mean, save_path='true_vs_measured.png')
    plot_error_distribution(error, save_path='error_distribution.png')

# 示例使用
if __name__ == "__main__":
    voxel_frames = np.random.randint(0, 2, (1, 256, 256, 13))  # 随机生成5帧voxel占据栅格
    print(voxel_frames==0)
    visualize_voxel_frames(voxel_frames)