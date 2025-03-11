# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import argparse
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from data.nuscenes_dataloader import DatasetSingleSeq
# from model import FeatEncoder, MotionNet, MotionNetMGDA
from utlis.model_utlis import *
from model.model import MotionNetMGDA, MotionNet, GeneralMotionNet, MixGeneralMotionNet, VAE_GeneralMotionNet
            # GridInsMotionNet
from model.LegoMotion import GridInsMotionNet
from model.PriorMotion import PriorMotionRaw
from model.en_model import En_MotionNet, GAN_MotionNet, GAN_MotionNetV2, En_GAN_MotionNet, En_GAN_MotionNetV2, GRUMotionNetVAE, UniGridInsGRUMotionNetVAE
# from model.model_ablation import
from model.model_ablation import UniGridGenMotionNet, MulUniGridInsGRUMotionNetVAE, MulUniGridInsGRUMotionNetVAE64x64, MulUniGridInsGRUMotionNetVAE32x32, GRUMotionNetVAEV2, \
                                MotionNet_CNN, MotionNet_GRU, MotionNetHigher
from model.be_sti import MotionNet as besti
from model.be_sti import GeneralMotionNet as Generalbesti
from model.en_gan_model import FeatEncoder, MotionNetGenerator

color_map = {0: "c", 1: "m", 2: "k", 3: "y", 4: "r"}
cat_names = {0: "bg", 1: "bus", 2: "ped", 3: "bike", 4: "other"}
'''
check whether the folder exists, if not, create one
'''
def check_folder(folder_path):
    if len(folder_path.split('/')) >= 2:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    else:
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
    return folder_path


def eval_motion_displacement(
    model_path=None,
    save_eval_file_path=None,
    use_adj_frame_pred=False,
    dataset_path=None,
    future_frame_skip=0,
    split="test",
    voxel_size=(0.25, 0.25, 0.4),
    area_extents=np.array([[-30.0, 30.0], [-30.0, 30.0], [-3.0, 2.0]]),
    num_future_sweeps=20,
    batch_size=1,
    num_workers=8,
    use_motion_state_pred_masking=False,
    height_feat_size=13,
    num_obj_category=5,
    which_model="MotionNet",
    general_flag=False,
    use_noise=False, noise_factor=0.1,
    sensor_type="nuscenes", args=None
):
    """
    Evaluate the motion prediction results.

    model_path: The path to the trained model
    save_eval_file_path: The path for saving the evaluation results
    use_adj_frame_pred: Whether to predict the relative offset between frames
    dataset_path: The path to the [val/test] dataset
    future_frame_skip: How many future frames need to be skipped within a contiguous sequence (ie, [1, 2, ... 20])
    split: [val/test]
    num_future_sweeps: The number of future frames
    batch_size: The batch size for [val/test] dataloader
    num_workers: The number of workers for dataloader
    use_motion_state_pred_masking: Whether to threshold the displacement predictions with predicted state estimation
    height_feat_size: The size along the height dimension
    num_obj_category: The number of object categories (including background)
    which_model: [MotionNet/MotionNetMGDA]
    """
    if (model_path is None) or (dataset_path is None):
        raise ValueError("Need to specify saved model path, log file save path and [val/test] dataset path.")
    if save_eval_file_path is None:
        warnings.warn("Did not specify the file path for saving the evaluation results.")

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DatasetSingleSeq(dataset_root=dataset_path, split=split, future_frame_skip=future_frame_skip,
                               use_velocity_flag=args.use_v,
                               voxel_size=voxel_size, area_extents=area_extents, num_category=num_obj_category,
                               sensor_type=sensor_type,
                               debug_mode=args.debug)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Specify the file for storing the evaluation results
    if save_eval_file_path is not None:
        logger_file = os.path.join(save_eval_file_path, 'eval_displacement.txt')
        save_eval_file_path = check_folder(save_eval_file_path)
        saver = open(logger_file, "w")

    # Load pre-trained network weights
    if which_model == 'MotionNet' or which_model == 'be-sti':
        # model = GeneralMotionNet(out_seq_len=10, motion_category_num=2, height_feat_size=height_feat_size)
        if args.use_v and args.use_vdir:
            if which_model == 'MotionNet':
                model = GeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 2)
            elif which_model == 'be-sti':
                model = Generalbesti(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 2)
        elif args.use_v and not args.use_vdir:
            if which_model == 'MotionNet':
                model = GeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 1)
            elif which_model == 'be-sti':
                model = Generalbesti(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 1)
        else:
            if which_model == 'MotionNet':
                model = MotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
            elif which_model == 'be-sti':
                model = besti(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)

        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    elif which_model == 'LEGOMotion':
        model = GridInsMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size, ins_query_num=95)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

    elif which_model == "MixGeneralMotionNet":
        if args.use_v and args.use_vdir:
            model = MixGeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size,
                                    with_direction=True)
        elif args.use_v and not args.use_vdir:
            model = MixGeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size,
                                    with_direction=False)
        else:
            model = MotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    elif which_model == "UniGenMotionNet":
        # Generative + velocity
        if args.use_v and args.use_vdir:
            model = VAE_GeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 2)
        elif args.use_v and not args.use_vdir:
            model = VAE_GeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 1)
        else:
            model = VAE_GeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    elif which_model == "GRUMotionNetVAE":
        if args.use_v and args.use_vdir:
            model =  GRUMotionNetVAE(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 2)
        elif args.use_v and not args.use_vdir:
            model =  GRUMotionNetVAE(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 1)
        else:
            model =  GRUMotionNetVAE(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    elif which_model == "UniGridGenMotionNet":
        if args.use_v and args.use_vdir:
            model =  UniGridGenMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 2)
        elif args.use_v and not args.use_vdir:
            model =  UniGridGenMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 1)
        else:
            model =  UniGridGenMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

    elif which_model == "UniGridInsGRUMotionNetVAE":
        model =  PriorMotionRaw(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

    elif which_model == "MulUniGridInsGRUMotionNetVAE64x64":
        model =  MulUniGridInsGRUMotionNetVAE64x64(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

    elif which_model == "MulUniGridInsGRUMotionNetVAE32x32":
        model = MulUniGridInsGRUMotionNetVAE32x32(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

    elif which_model == "MotionNet_CNN":
        model = MotionNet_CNN(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

    elif which_model == "MotionNetHigher":
        model = MotionNetHigher(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

    elif which_model == "MotionNet_GRU":
        model = MotionNet_GRU(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

    elif which_model == "MotionNetMGDA":  # MotionNetMGDA
        model_encoder = FeatEncoder()
        model_head = MotionNetMGDA(out_seq_len=20, motion_category_num=2)

        model_encoder = nn.DataParallel(model_encoder)
        model_head = nn.DataParallel(model_head)

        checkpoint = torch.load(model_path)
        model_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model_head.load_state_dict(checkpoint['head_state_dict'])

        model_encoder = model_encoder.to(device)
        model_head = model_head.to(device)
    elif which_model == "En_MotionNet":
        model = En_MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=height_feat_size)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    elif which_model == "GAN_MotionNet":
        model = GAN_MotionNet(out_seq_len=20, motion_category_num=2)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    elif which_model == "GAN_MotionNetV2":
        model = GAN_MotionNetV2(out_seq_len=20, motion_category_num=2)
        # model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    elif which_model == "En_GAN_MotionNet":
        model = En_GAN_MotionNet(out_seq_len=20, motion_category_num=2)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    elif which_model == "En_GAN_MotionNetV2":
        model = En_GAN_MotionNetV2(out_seq_len=20, motion_category_num=2)
        model = nn.DataParallel(model)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    elif which_model == "MotionNetGAN":
        model = FeatEncoder()
        generator = MotionNetGenerator()
        model = nn.DataParallel(model)
        generator = nn.DataParallel(generator)
        checkpoint = torch.load(model_path)
        # breakpoint()
        model.load_state_dict(checkpoint['model_state_dict'])
        generator.load_state_dict(checkpoint['generator_model_state_dict'])
        model = model.to(device=device)
        generator = generator.to(device=device)

    # The speed intervals for grouping the cells
    # speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0]])  # unit: m/s
    # We do not consider > 20m/s, since objects in nuScenes appear inside city and rarely exhibit very high speed
    speed_intervals = np.array([[0.0, 0.0], [0, 5.0], [5.0, 20.0]])
    selected_future_sweeps = np.arange(
        0, num_future_sweeps + 1, future_frame_skip + 1
    )  # We evaluate predictions at [0.2, 0.4, ..., 1]s
    selected_future_sweeps = selected_future_sweeps[1:]
    last_future_sweep_id = selected_future_sweeps[-1]
    distance_intervals = speed_intervals * (last_future_sweep_id / num_future_sweeps)  # "20" is because the LIDAR scanner is 20Hz

    cell_groups = list()  # grouping the cells with different speeds
    for i in range(distance_intervals.shape[0]):
        cell_statistics = list()

        for j in range(len(selected_future_sweeps)):
            # corresponds to each row, which records the MSE, median etc.
            cell_statistics.append([])
        cell_groups.append(cell_statistics)

    # Make prediction
    model.eval()
    # if which_model == ('MotionNet' or 'En_MotionNet') or which_model == 'be-sti' or which_model == 'UniGenMotionNet' \
    #                 or which_model == 'MixGeneralMotionNet' or which_model == "GRUMotionNetVAE":
    #     model.eval()
    # elif which_model == "UniGridGenMotionNet":
    #     model.eval()
    # elif which_model == "UniInsGridGenMotionNet":
    #     model.eval()

    # elif which_model == "MulUniGridInsGRUMotionNetVAE64x64" or which_model == "MulUniGridInsGRUMotionNetVAE32x32":
    #     model.eval()
    # elif which_model == "MotionNet_CNN" or which_model == "MotionNetHigher" or which_model == "MotionNet_GRU":
    #     model.eval()
    # elif which_model == "LEGOMotion":
    #     model.eval()
    # else:
    #     model_encoder.eval()
    #     model_head.eval()

    pixel_acc = 0  # for computing mean pixel classification accuracy
    overall_cls_pred = list()  # to compute classification accuracy for each object category
    overall_cls_gt = list()  # to compute classification accuracy for each object category

    for i, data in enumerate(dataloader, 0):
        if sensor_type == "nuscenes" or sensor_type == "at128_0" or not args.use_v:
            padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, \
                non_empty_map, pixel_cat_map_gt, pixel_instance_map_gt, past_steps, future_steps, motion_gt = data

        elif sensor_type == "wt128_0" and args.use_v:
            padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, \
                non_empty_map, pixel_cat_map_gt, pixel_instance_map_gt, past_steps, future_steps, motion_gt, \
                all_pixel_velocity_maps, all_pixel_direction_maps, all_pixel_cos_maps = data

            all_pixel_velocity_maps = all_pixel_velocity_maps.to(device)
            all_pixel_direction_maps = all_pixel_direction_maps.to(device)

        padded_voxel_points = padded_voxel_points.to(device)
        if general_flag:
            # segment
            pixel_cat_map_gt_numpy = pixel_cat_map_gt.numpy()
            non_empty_map_numpy = non_empty_map.numpy()
            # class_pred_numpy = class_pred.cpu().numpy()

            # Convert the category map
            max_prob = np.amax(pixel_cat_map_gt_numpy, axis=-1)
            filter_mask = max_prob == 1.0  # Note: some of the cell probabilities are soft probabilities
            pixel_cat_map_numpy = np.argmax(pixel_cat_map_gt_numpy, axis=-1) + 1  # category starts from 1 (background), etc

            # Convert category label to nongernal label
            cat_nongeneral_map = np.zeros_like(pixel_cat_map_numpy)
            cat_nongeneral_map[pixel_cat_map_numpy == 5] = 1

        noise:torch.tensor = noise_sample_ACGAN(batch_size=padded_voxel_points.shape[0], n_z=32, n_size=256).to(device)
        if use_noise:
            noise = noise * noise_factor
        else:
            noise = noise * 0.0

        with torch.no_grad():
            with torch.no_grad():
                if which_model == ('MotionNet' or 'En_MotionNet') or which_model == 'be-sti' \
                    or which_model == 'MixGeneralMotionNet' or which_model == "LEGOMotion":
                    # disp_pred, class_pred, motion_pred = model(padded_voxel_points, None, None)
                    if args.use_v and args.use_vdir:
                        disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points, all_pixel_velocity_maps, all_pixel_direction_maps)
                    elif args.use_v and not args.use_vdir:
                        disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points, all_pixel_velocity_maps)
                    else:
                        disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points)
                elif which_model == "UniGenMotionNet":
                    if args.use_v and args.use_vdir:
                        disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points, prior_velocity=all_pixel_velocity_maps, prior_direction=all_pixel_direction_maps)
                    elif args.use_v and not args.use_vdir:
                        disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points, all_pixel_velocity_maps)
                    else:
                        disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points)
                elif which_model == "GRUMotionNetVAE":
                    if args.use_v and args.use_vdir:
                        disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points, prior_velocity=all_pixel_velocity_maps, prior_direction=all_pixel_direction_maps)
                    elif args.use_v and not args.use_vdir:
                        disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points, all_pixel_velocity_maps)
                    else:
                        disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points)

                elif which_model == "UniGridGenMotionNet":
                    if args.use_v and args.use_vdir:
                        disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points, prior_velocity=all_pixel_velocity_maps, prior_direction=all_pixel_direction_maps)
                    elif args.use_v and not args.use_vdir:
                        disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points, all_pixel_velocity_maps)
                    else:
                        disp_pred, class_pred, motion_pred, _, _ = model(padded_voxel_points)

                elif which_model == "UniInsGridGenMotionNet":
                    if args.use_v and args.use_vdir:
                        disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points, prior_velocity=all_pixel_velocity_maps, prior_direction=all_pixel_direction_maps)
                    elif args.use_v and not args.use_vdir:
                        disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points, all_pixel_velocity_maps)
                    else:
                        disp_pred, class_pred, motion_pred, _, _ = model(padded_voxel_points)
                elif which_model == "UniGridInsGRUMotionNetVAE":
                    disp_pred, class_pred, motion_pred, _, _ = model(padded_voxel_points, training_only=False)

                elif which_model == "MulUniGridInsGRUMotionNetVAE64x64" or which_model == "MulUniGridInsGRUMotionNetVAE32x32":
                    if args.use_v and args.use_vdir:
                        disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points, prior_velocity=all_pixel_velocity_maps, prior_direction=all_pixel_direction_maps)
                    elif args.use_v and not args.use_vdir:
                        disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points, all_pixel_velocity_maps)
                    else:
                        disp_pred, class_pred, motion_pred, _, _ = model(padded_voxel_points)

                elif which_model == "MotionNet_CNN" or which_model == "MotionNetHigher" or which_model == "MotionNet_GRU":
                    if args.use_v and args.use_vdir:
                        disp_pred, class_pred, motion_pred = model(padded_voxel_points, all_pixel_velocity_maps, all_pixel_direction_maps)
                    elif args.use_v and not args.use_vdir:
                        disp_pred, class_pred, motion_pred = model(padded_voxel_points, all_pixel_velocity_maps)
                    else:
                        disp_pred, class_pred, motion_pred = model(padded_voxel_points)

                elif which_model in ['GAN_MotionNet', 'En_GAN_MotionNet']:
                    disp_pred, class_pred, motion_pred, aux_disp_pred, aux_class_pred, aux_motion_pred = model(padded_voxel_points,noise)
                elif which_model in ['GAN_MotionNetV2', 'En_GAN_MotionNetV2']:
                    disp_pred, class_pred, motion_pred, aux_disp_pred, aux_class_pred, aux_motion_pred = model(padded_voxel_points,noise)
                elif which_model == "MotionNetGAN":
                    # Make prediction, aux prediction
                    feat = model(padded_voxel_points
                                )

                    disp_pred, class_pred, motion_pred, \
                        aux_disp_pred, aux_class_pred, aux_motion_pred = generator(feat, noise)
                else:
                    shared_feats = model_encoder(padded_voxel_points)
                    disp_pred, class_pred, motion_pred = model_head(shared_feats)

            pred_shape = disp_pred.size()
            disp_pred = disp_pred.view(all_disp_field_gt.size(0), -1, pred_shape[-3], pred_shape[-2], pred_shape[-1])
            disp_pred = disp_pred.contiguous()
            disp_pred = disp_pred.cpu().numpy()

            if use_adj_frame_pred:
                for c in range(1, disp_pred.shape[1]):
                    disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]

            if use_motion_state_pred_masking:
                motion_pred_numpy = motion_pred.cpu().numpy()
                motion_pred_numpy = np.argmax(motion_pred_numpy, axis=1)
                mask = motion_pred_numpy == 0


                class_pred_numpy = class_pred.cpu().numpy()
                class_pred_cat = np.argmax(class_pred_numpy, axis=1)
                class_mask = class_pred_cat == 0  # background mask

                # For those with very small movements, we consider them as static
                last_pred = disp_pred[:, -1, :, :, :]
                last_pred_norm = np.linalg.norm(last_pred, ord=2, axis=1)  # out: (batch, h, w)
                thd_mask = last_pred_norm <= 0.2

                cat_weight_map = np.ones_like(class_pred_cat, dtype=np.float32)
                cat_weight_map[mask] = 0.0
                cat_weight_map[class_mask] = 0.0
                cat_weight_map[thd_mask] = 0.0
                motion_masks = None
                if motion_masks is None:
                    motion_masks = motion_gt
                    motion_masks = motion_masks.permute(0, 3, 1, 2)
                    motion_mask_numpy = motion_masks.cpu().numpy()
                    motion_mask_numpy = np.argmax(motion_mask_numpy, axis=1)
                    # breakpoint()
                    mask_stat = motion_mask_numpy == 0
                    mask_dyn = motion_mask_numpy == 1
                    mask_dyn = mask_dyn[:, None, None, ...]
                    # disp_pred = disp_pred * mask_dyn
                    cat_weight_map[mask_stat] = 0.0

                cat_weight_map = cat_weight_map[:, np.newaxis, np.newaxis, ...]  # (batch, 1, 1, h, w)

                disp_pred = disp_pred * cat_weight_map
                if general_flag:
                    cat_nongeneral_map = cat_nongeneral_map[:, np.newaxis, np.newaxis, ...] # (batch, 1, 1, h, w)
                    disp_pred = disp_pred * cat_nongeneral_map

        # Pre-processing
        all_disp_field_gt = all_disp_field_gt.numpy()  # (bs, seq, h, w, channel)
        future_steps = future_steps.numpy()[0]

        valid_pixel_maps = all_valid_pixel_maps[:, -future_steps:, ...].contiguous()
        valid_pixel_maps = valid_pixel_maps.numpy()

        all_disp_field_gt = all_disp_field_gt[
            :,
            -future_steps:,
        ]
        all_disp_field_gt = np.transpose(all_disp_field_gt, (0, 1, 4, 2, 3))
        all_disp_field_gt_norm = np.linalg.norm(all_disp_field_gt, ord=2, axis=2)

        # -----------------------------------------------------------------------------------
        # Compute the evaluation metrics
        # First, compute the displacement prediction error;
        # Compute the static and moving cell masks, and
        # Iterate through the distance intervals and group the cells based on their speeds;
        upper_thresh = 0.2
        upper_bound = (future_frame_skip + 1) / out_seq_len * upper_thresh

        static_cell_mask = all_disp_field_gt_norm <= upper_bound
        static_cell_mask = np.all(static_cell_mask, axis=1)  # along the temporal axis
        moving_cell_mask = np.logical_not(static_cell_mask)
        # breakpoint()
        for j, d in enumerate(distance_intervals):
            for slot, s in enumerate((selected_future_sweeps - 1)):  # selected_future_sweeps: [4, 8, ...]
                curr_valid_pixel_map = valid_pixel_maps[:, s]

                if j == 0:  # corresponds to static cells
                    curr_mask = np.logical_and(curr_valid_pixel_map, static_cell_mask)
                else:
                    # We use the displacement between keyframe and the last sample frame as metrics
                    last_gt_norm = all_disp_field_gt_norm[:, -1]
                    mask = np.logical_and(d[0] <= last_gt_norm, last_gt_norm < d[1])

                    curr_mask = np.logical_and(curr_valid_pixel_map, mask)
                    curr_mask = np.logical_and(curr_mask, moving_cell_mask)

                # Since in nuScenes (with 32-line LiDAR) the points (cells) in the distance are very sparse,
                # we evaluate the performance for cells within the range [-30m, 30m] along both x, y dimensions.
                border = 8
                roi_mask = np.zeros_like(curr_mask, dtype=bool)
                roi_mask[:, border:-border, border:-border] = True
                curr_mask = np.logical_and(curr_mask, roi_mask)

                cell_idx = np.where(curr_mask == True)

                if general_flag:
                    all_disp_field_gt = all_disp_field_gt * cat_nongeneral_map
                gt = all_disp_field_gt[:, s]
                pred = disp_pred[:, s]
                norm_error = np.linalg.norm(gt - pred, ord=2, axis=1)

                cell_groups[j][slot].append(norm_error[cell_idx])

        # -----------------------------------------------------------------------------------
        # Second, compute the classification accuracy
        pixel_cat_map_gt_numpy = pixel_cat_map_gt.numpy()
        non_empty_map_numpy = non_empty_map.numpy()
        class_pred_numpy = class_pred.cpu().numpy()

        # Convert the category map
        max_prob = np.amax(pixel_cat_map_gt_numpy, axis=-1)
        filter_mask = max_prob == 1.0  # Note: some of the cell probabilities are soft probabilities
        pixel_cat_map_numpy = np.argmax(pixel_cat_map_gt_numpy, axis=-1) + 1  # category starts from 1 (background), etc
        pixel_cat_map_numpy = (pixel_cat_map_numpy * non_empty_map_numpy * filter_mask).astype(int)

        class_pred_numpy = np.transpose(class_pred_numpy, (0, 2, 3, 1))
        class_pred_numpy = np.argmax(class_pred_numpy, axis=-1) + 1
        class_pred_numpy = (class_pred_numpy * non_empty_map_numpy * filter_mask).astype(int)

        border = 8
        roi_mask = np.zeros_like(non_empty_map_numpy)
        roi_mask[:, border:-border, border:-border] = 1.0

        tmp = pixel_cat_map_numpy == class_pred_numpy
        denominator = np.sum(non_empty_map_numpy * filter_mask * roi_mask)
        pixel_acc += np.sum(tmp * non_empty_map_numpy * filter_mask * roi_mask) / denominator

        # For computing confusion matrix, in order to compute classification accuracy for each category
        count_mask = non_empty_map_numpy * filter_mask * roi_mask
        idx_fg = np.where(count_mask > 0)

        overall_cls_gt.append(pixel_cat_map_numpy[idx_fg])
        overall_cls_pred.append(class_pred_numpy[idx_fg])
        if (i+1) % 100 == 0:
            print("Finish sample [{}/{}]".format(i + 1, int(np.ceil(len(dataset) / float(batch_size)))))

    # Compute the statistics
    dump_res = []

    # Compute the statistics of displacement prediction error
    for i, d in enumerate(speed_intervals):
        group = cell_groups[i]
        print("--------------------------------------------------------------")
        print("For cells within speed range [{}, {}]:\n".format(d[0], d[1]))
        if save_eval_file_path is not None:
            saver.write("--------------------------------------------------------------\n")
            saver.write("For cells within speed range [{}, {}]:\n\n".format(d[0], d[1]))

        dump_error = []
        dump_error_quantile_50 = []

        for s in range(len(selected_future_sweeps)):
            row = group[s]

            errors = np.concatenate(row) if len(row) != 0 else row

            if len(errors) == 0:
                mean_error = None
                error_quantile_50 = None
            else:
                mean_error = np.average(errors)
                error_quantile_50 = np.quantile(errors, 0.5)

            dump_error.append(mean_error)
            dump_error_quantile_50.append(error_quantile_50)

            msg = "Frame {}:\nThe mean error is {}\nThe 50% error quantile is {}".format(
                selected_future_sweeps[s], mean_error, error_quantile_50
            )
            print(msg)
            if save_eval_file_path is not None:
                saver.write(msg + "\n")
                saver.flush()
        print("--------------------------------------------------------------\n")
        if save_eval_file_path is not None:
            saver.write("--------------------------------------------------------------\n\n")

        dump_res.append(dump_error + dump_error_quantile_50)

    # Compute the statistics of mean pixel classification accuracy
    pixel_acc = pixel_acc / len(dataset)
    print("Mean pixel classification accuracy: {}".format(pixel_acc))
    if save_eval_file_path is not None:
        saver.write("Mean pixel classification accuracy: {}\n".format(pixel_acc))

    # Compute the mean classification accuracy for each object category
    overall_cls_gt = np.concatenate(overall_cls_gt)
    overall_cls_pred = np.concatenate(overall_cls_pred)

    cm = confusion_matrix(overall_cls_gt, overall_cls_pred)
    cm_sum = np.sum(cm, axis=1)
    cls_num = len(np.unique(overall_cls_gt))
    # try:
    #     print("The number of cls pred:{}".format(len(np.unique(overall_cls_pred))))
    #     mean_cat = cm[np.arange(cls_num), np.arange(cls_num)] / cm_sum

    # except:
        # cls_num = np.min([len(np.unique(overall_cls_gt)), len(np.unique(overall_cls_pred))])
    cls_num = cm.shape[0]
    # breakpoint()
    mean_cat = cm[np.arange(cls_num), np.arange(cls_num)] / cm_sum
    # breakpoint()
    cat_map = {0: "Bg", 1: "Vehicle", 2: "Ped", 3: "Bike", 4: "Others"}
    for i in range(len(mean_cat)):
        print("mean cat accuracy of {}: {}".format(cat_map[i], mean_cat[i]))
    print("mean instance acc: ", np.mean(mean_cat))
    if save_eval_file_path is not None:
        for i in range(len(mean_cat)):
            saver.write("mean cat accuracy of {}: {}\n".format(cat_map[i], mean_cat[i]))
        saver.write("mean instance acc: {}\n".format(np.mean(mean_cat)))

    # Dump the displacement prediction statistics into a column, which is convenient to copy-paste the results into sheets
    if save_eval_file_path is not None:
        dump_res = np.concatenate(dump_res)
        saver.write("\nReshape the displacement prediction statistics into a column:\n")
        saver.write("\n".join(str(item) for item in dump_res))
        saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default=None, type=str, help="The path to the [val/test] dataset")
    parser.add_argument("-m", "--model", default=None, type=str, help="The path to the trained model")
    parser.add_argument("-l", "--log", default=None, type=str, help="The path to the txt file for saving eval results")
    parser.add_argument("-s", "--split", default="test", type=str, help="Which split [val/test]")
    parser.add_argument("-b", "--bs", default=1, type=int, help="Batch size")
    parser.add_argument("-w", "--worker", default=8, type=int, help="The number of workers")
    parser.add_argument("-n", "--net", default="MotionNet", type=str, help="Which network [MotionNet/MotionNetMGDA]")
    parser.add_argument("-a", "--adj", action="store_false", help="Whether predict the relative offset between frames")
    parser.add_argument("-j", "--jitter", action="store_false", help="Whether to apply jitter suppression")
    parser.add_argument('--gpu', default='0', type=str, help='use gpu id')
    parser.add_argument('-g','--general', action='store_true', help='Whether to apply general')
    parser.add_argument('--noise', action='store_true', help='Whether to apply noise suppression')
    parser.add_argument('--noise_factor', default=0.1, type=float, help='The noise factor')

    parser.add_argument('--sensor_type', default='nuscenes', type=str, help='The dataset')
    parser.add_argument('--use_v', action='store_true', help='whether to apply velocity')
    parser.add_argument('--use_vdir', action='store_true', help='whether to apply velocity direction')
    parser.add_argument('--debug', action='store_true', help='Switch to debug mode')
    args = parser.parse_args()
    print(args)

    if args.sensor_type == 'wt128_0' or args.sensor_type == 'at128_0':
        out_seq_len = 10
    else:
        out_seq_len = 20

    area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
    if args.sensor_type == 'wt128_0':
        area_extents = np.array([[0., 64.], [-32., 32.], [-2., 3.]])
    eval_motion_displacement(
        model_path=args.model,
        save_eval_file_path=args.log,
        split=args.split,
        dataset_path=args.data,
        which_model=args.net,
        batch_size=args.bs,
        num_workers=args.worker,
        voxel_size=(0.25, 0.25, 0.4),
        # area_extents=np.array([[-32.0, 32.0], [-32.0, 32.0], [-3.0, 2.0]]),
        area_extents=area_extents,
        future_frame_skip=0,
        num_future_sweeps=out_seq_len,
        use_adj_frame_pred=args.adj,
        use_motion_state_pred_masking=args.jitter,
        height_feat_size=13,
        num_obj_category=5,
        general_flag=args.general,
        sensor_type=args.sensor_type,
        args=args,
    )
