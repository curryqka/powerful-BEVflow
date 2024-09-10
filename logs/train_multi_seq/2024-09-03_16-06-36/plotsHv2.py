import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from data.data_utils import DataStatistic

color_map = {0: 'c', 1: 'm', 2: 'k', 3: 'y', 4: 'r'}
height_feat_size=13
cat_names = {0: 'bg', 1: 'bus', 2: 'ped', 3: 'bike', 4: 'other'}

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def vis_model_per_sample_data(bev_input_data, gan_bev_input_data, frame_skip=3, voxel_size=(0.25, 0.25, 0.4),
                              loaded_models=None, which_model="MotionNet", model_path=None, img_save_dir=None,
                              use_adj_frame_pred=False, use_motion_state_pred_masking=False, frame_idx=0, disp=True):
    all_disp_field_gt, pixel_cat_map_gt, non_empty_map = bev_input_data
    disp_pred, cat_pred, pred_non_empty_map = gan_bev_input_data
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))


    # We only show the cells having one-hot category vectors
    max_prob = np.amax(pixel_cat_map_gt, axis=-1)
    filter_mask = max_prob == 1.0
    pixel_cat_map = np.argmax(pixel_cat_map_gt, axis=-1) + 1  # category starts from 1 (background), etc
    pixel_cat_map = (pixel_cat_map * non_empty_map * filter_mask).astype(int)

    cat_pred = np.argmax(cat_pred, axis=-1) + 1
    cat_pred = (cat_pred * non_empty_map * filter_mask).astype(int)
    # ...
    # Draw the LIDAR and quiver plots
    # The distant points are very sparse and not reliable. We do not show them.
    border_meter = 4
    border_pixel = border_meter * 4
    x_lim = [-(32 - border_meter), (32 - border_meter)]
    y_lim = [-(32 - border_meter), (32 - border_meter)]
    # --- Visualization ---

    # Plot quiver. We only show non-empty vectors. Plot each category.
    field_gt = all_disp_field_gt[-1]
    idx_x = np.arange(field_gt.shape[0])
    idx_y = np.arange(field_gt.shape[1])
    idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
    qk = [None] * len(color_map)  # for quiver key

    
    for k in range(len(color_map)):
        # ------------------------ Ground-truth ------------------------
        mask = pixel_cat_map == (k + 1)

        # For cells with very small movements, we threshold them to be static
        field_gt_norm = np.linalg.norm(field_gt, ord=2, axis=-1)  # out: (h, w)
        thd_mask = field_gt_norm <= 0.4
        field_gt[thd_mask, :] = 0

        # Get the displacement field
        X = idx_x[mask]
        Y = idx_y[mask]
        U = field_gt[:, :, 0][mask] / voxel_size[0]  # the distance between pixels is w.r.t. grid size (e.g., 0.2m)
        V = field_gt[:, :, 1][mask] / voxel_size[1]

        qk[k] = ax[0].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color=color_map[k])
        ax[0].quiverkey(qk[k], X=0.0 + k/5.0, Y=1.1, U=20, label=cat_names[k], labelpos='E')
        ax[0].set_xlim(border_pixel, field_gt.shape[0] - border_pixel)
        ax[0].set_ylim(border_pixel, field_gt.shape[1] - border_pixel)
        ax[0].set_aspect('equal')
        ax[0].title.set_text('Ground-truth')
        ax[0].axis('off')
        
        
        # ------------------------ Prediction ------------------------
        # Show the prediction results. We show the cells corresponding to the non-empty one-hot gt cells.
        mask_pred = cat_pred == (k + 1)
        field_pred = disp_pred[-1]  # Show last prediction, ie., the 20-th frame

        # For cells with very small movements, we threshold them to be static
        field_pred_norm = np.linalg.norm(field_pred, ord=2, axis=-1)  # out: (h, w)
        thd_mask = field_pred_norm <= 0.4
        field_pred[thd_mask, :] = 0

        # We use the same indices as the ground-truth, since we are currently focused on the foreground
        X_pred = idx_x[mask_pred]
        Y_pred = idx_y[mask_pred]
        U_pred = field_pred[:, :, 0][mask_pred] / voxel_size[0]
        V_pred = field_pred[:, :, 1][mask_pred] / voxel_size[1]

        ax[1].quiver(X_pred, Y_pred, U_pred, V_pred, angles='xy', scale_units='xy', scale=1, color=color_map[k])
        ax[1].set_xlim(border_pixel, field_pred.shape[0] - border_pixel)
        ax[1].set_ylim(border_pixel, field_pred.shape[1] - border_pixel)
        ax[1].set_aspect('equal')
        ax[1].title.set_text('Prediction')
        ax[1].axis('off')
        
    img_save_dir = check_folder(img_save_dir)
    print("finish sample {}".format(frame_idx))
    plt.savefig(os.path.join(img_save_dir, str(frame_idx) + '.png'))

    if disp:
        plt.pause(0.02)
    # Clear only the active axes
    ax[0].clear()
    ax[1].clear()

    if disp:
        plt.show()

def load_bev_data(sample_file):
    gt_data_handle = np.load(sample_file, allow_pickle=True)
    gt_dict = gt_data_handle.item()

    print("processing file: {}".format(sample_file))
 
    # read the gt_dict
    dims = gt_dict['3d_dimension']
    num_future_pcs = gt_dict['num_future_pcs']
    num_past_pcs = gt_dict['num_past_pcs']
    pixel_indices = gt_dict['pixel_indices']

    padded_voxel_points_list = list()
    all_disp_field_gt_list = list()
    all_valid_pixel_maps_list = list()
    non_empty_map_list = list()
    pixel_cat_map_list = list()
    # trans_matrices_list = list()
    pixel_instance_map_list = list()

    dims = gt_dict['3d_dimension']
    num_future_pcs = gt_dict['num_future_pcs']
    pixel_indices = gt_dict['pixel_indices']
    # trans_matrices = gt_dict['trans_matrices']

    sparse_disp_field_gt = gt_dict['disp_field']
    all_disp_field_gt = np.zeros((num_future_pcs, dims[0], dims[1], 2), dtype=np.float32)
    all_disp_field_gt[:, pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_disp_field_gt[:]

    sparse_valid_pixel_maps = gt_dict['valid_pixel_map']
    all_valid_pixel_maps = np.zeros((num_future_pcs, dims[0], dims[1]), dtype=np.float32)
    all_valid_pixel_maps[:, pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_valid_pixel_maps[:]

    sparse_pixel_cat_maps = gt_dict['pixel_cat_map']
    pixel_cat_map = np.zeros((dims[0], dims[1], 5), dtype=np.float32)
    pixel_cat_map[pixel_indices[:, 0], pixel_indices[:, 1], :] = sparse_pixel_cat_maps[:]

    sparse_pixel_instance_maps = gt_dict['pixel_instance_ids']
    pixel_instance_map = np.zeros((dims[0], dims[1]), dtype=np.uint8)
    pixel_instance_map[pixel_indices[:, 0], pixel_indices[:, 1]] = sparse_pixel_instance_maps[:]

    non_empty_map = np.zeros((dims[0], dims[1]), dtype=np.float32)
    non_empty_map[pixel_indices[:, 0], pixel_indices[:, 1]] = 1.0

    padded_voxel_points = list()
    num_past_pcs = 5
    for i in range(num_past_pcs):
        indices = gt_dict['voxel_indices_' + str(i)]
        # 20230302 np.bool --> bool
        curr_voxels = np.zeros(dims, dtype=bool)
        curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
        padded_voxel_points.append(curr_voxels)
    padded_voxel_points = np.stack(padded_voxel_points, 0).astype(np.float32)

    padded_voxel_points_list.append(padded_voxel_points[-5:])
    all_disp_field_gt_list.append(all_disp_field_gt)
    all_valid_pixel_maps_list.append(all_valid_pixel_maps)
    non_empty_map_list.append(non_empty_map)
    pixel_cat_map_list.append(pixel_cat_map)
    # trans_matrices_list.append(trans_matrices)
    pixel_instance_map_list.append(pixel_instance_map)

    padded_voxel_points_list = np.stack(padded_voxel_points_list, 0)
    all_disp_field_gt_list = np.stack(all_disp_field_gt_list, 0)
    all_valid_pixel_maps_list = np.stack(all_valid_pixel_maps_list, 0)
    non_empty_map_list = np.stack(non_empty_map_list, 0)
    pixel_cat_map_list = np.stack(pixel_cat_map_list, 0)
    # trans_matrices_list = np.stack(trans_matrices_list, 0)
    pixel_instance_map_list = np.stack(pixel_instance_map_list, 0)
    
    return all_disp_field_gt, pixel_cat_map, non_empty_map
def visual_gan(dataset_root, gan_dataset_root, model_path=None, save_dir="output"):
    data_statistic = DataStatistic(dataset_root=dataset_root, save_path='./data',
                                    )
    gan_data_statistic = DataStatistic(dataset_root=gan_dataset_root, save_path='./data',
                                    )
    # load the GeneratorDisp model
    # model = GeneratorDisp(input_channel=100)
    # model = nn.DataParallel(model)
    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint['modelG_state_dict'])
    # model = model.to(device)
    # model.eval()
    
    save_data_list = list()
    seq_file_idx_list = list()
    
    # for sample_file, gan_sample_file in zip(data_statistic.seq_files, gan_data_statistic.seq_files):
    for sample_file in data_statistic.seq_files:
        for gan_sample_file in gan_data_statistic.seq_files:
            seq_idx = sample_file.split("/")[-2]
            gan_seq_idx = gan_sample_file.split("/")[-2]
            if seq_idx == gan_seq_idx:
                all_disp_field_gt, pixel_cat_map, non_empty_map = load_bev_data(sample_file=sample_file)
                pred_disp_field, pred_pixel_cat_map, pred_non_empty_map = load_bev_data(sample_file=gan_sample_file)
                bev_input_data = (all_disp_field_gt, pixel_cat_map, non_empty_map)
                gan_bev_input_data = (pred_disp_field, pred_pixel_cat_map, pred_non_empty_map)
                vis_model_per_sample_data(bev_input_data, gan_bev_input_data, frame_skip=3, voxel_size=(0.25, 0.25, 0.4),
                                    loaded_models=None, which_model="MotionNet", model_path=model_path, img_save_dir=save_dir,
                                    use_adj_frame_pred=True, use_motion_state_pred_masking=True, frame_idx=0, disp=True)

if __name__ == "__main__":
    visual_gan(dataset_root="/mnt/personal/qiankangan/at128_0/train/",gan_dataset_root="/mnt/personal/qiankangan/at128_0/train/")