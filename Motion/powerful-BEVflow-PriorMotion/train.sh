# -*- coding: utf-8 -*-

# 这样也行
# coding:utf-8
# export PYTHONPATH=/home/qiankangan/UniGenMotionNet:$PYTHONPATH
# export PYTHONPATH=/home/qiankangan/UniGenMotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH
# sensor_type=$1
# GPU=$2
# OMP_NUM_THREADS=1 python train_VAE.py --data /mnt/personal/qiankangan/input-data --evaldata /mnt/personal/qiankangan/val \
# --batch 2 --nepoch 45 --nworker 4 --lr 0.0016 \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
# --log --auxiliary --use_adj_frame_pred --model GRUVAE_MotionNet \
# --logpath /mnt/personal/qiankangan/logs/GRUVAE_bs2_MotionNet --general --gpu $GPU \
# --sensor_type $sensor_type

# # origin
# OMP_NUM_THREADS=1 python train_VAE.py --data /mnt/personal/qiankangan/input-data --evaldata /mnt/personal/qiankangan/val \
# --batch 4 --nepoch 45 --nworker 4 --lr 0.0016 \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
# --log --auxiliary --use_adj_frame_pred --model GRUVAE_MotionNet \
# --logpath /mnt/personal/qiankangan/logs/GRUVAE_origin_bs4_MotionNet --gpu 4 \
# --sensor_type nuscenes

# # uniGen: grid+ins prior + GRU
# OMP_NUM_THREADS=1 python train_VAE.py --data /mnt/personal/qiankangan/input-data --evaldata /mnt/personal/qiankangan/val \
# --batch 4 --nepoch 45 --nworker 4 --lr 0.0016 \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
# --log --auxiliary --use_adj_frame_pred --model UniGRUVAE_MotionNet \
# --logpath /mnt/personal/qiankangan/logs/UniGRUVAE_bs4_MotionNet --general --gpu 7 \
# --sensor_type nuscenes --resume /mnt/personal/qiankangan/logs/UniGRUVAE_bs4_MotionNet/train_multi_seq/2024-11-04_22-02-43

# # Final model
# OMP_NUM_THREADS=1 python train_VAE.py --data /mnt/personal/qiankangan/input-data --evaldata /mnt/personal/qiankangan/val \
# --batch 4 --nepoch 45 --nworker 4 --lr 0.0016 \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
# --log --auxiliary --use_adj_frame_pred --model UniGridInsGRUMotionNetVAE \
# --logpath /mnt/personal/qiankangan/logs/UniInsGridGRUVAE_bs4_MotionNet --general --gpu 3 \
# --sensor_type nuscenes

# # ------------ablation-----------------
# #  ablation grid info
# OMP_NUM_THREADS=1 python train_ablation.py --data /mnt/personal/qiankangan/input-data --evaldata /mnt/personal/qiankangan/val \
# --batch 4 --nepoch 45 --nworker 4 --lr 0.0016 \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
# --log --auxiliary --use_adj_frame_pred --model UniGridGenMotionNet \
# --logpath /mnt/personal/qiankangan/logs/UniGridGenMotionNet_bs4_ablation --general --gpu 3 \
# --sensor_type nuscenes

# # add feat selection
# OMP_NUM_THREADS=1 python train_ablation.py --data /mnt/personal/qiankangan/input-data --evaldata /mnt/personal/qiankangan/val \
# --batch 4 --nepoch 45 --nworker 4 --lr 0.0016 \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
# --log --auxiliary --use_adj_frame_pred --model MulUniGridInsGRUMotionNetVAE \
# --logpath /mnt/personal/qiankangan/logs/MulUniGridInsGRUMotionNetVAE_bs4_ablation --general --gpu 3 \
# --sensor_type nuscenes --grid_prior

# # 32x32
# OMP_NUM_THREADS=1 python train_ablation.py --data /mnt/personal/qiankangan/input-data --evaldata /mnt/personal/qiankangan/val \
# --batch 4 --nepoch 45 --nworker 4 --lr 0.0016 \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
# --log --auxiliary --use_adj_frame_pred --model MulUniGridInsGRUMotionNetVAE32x32 \
# --logpath /mnt/personal/qiankangan/logs/MulUniGridInsGRUMotionNetVAE32x32_bs4_ablation --general --gpu 0 \
# --sensor_type nuscenes --grid_prior

# # 64x64
# OMP_NUM_THREADS=1 python train_ablation.py --data /mnt/personal/qiankangan/input-data --evaldata /mnt/personal/qiankangan/val \
# --batch 4 --nepoch 45 --nworker 4 --lr 0.0016 \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
# --log --auxiliary --use_adj_frame_pred --model MulUniGridInsGRUMotionNetVAE64x64 \
# --logpath /mnt/personal/qiankangan/logs/MulUniGridInsGRUMotionNetVAE64x64_bs4_ablation --general --gpu 3 \
# --sensor_type nuscenes --grid_prior

# data aug
# OMP_NUM_THREADS=1 python train_VAE.py --data /mnt/personal/qiankangan/input-data --evaldata /mnt/personal/qiankangan/val \
# --batch 2 --nepoch 45 --nworker 4 --lr 0.0016 \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
# --log --auxiliary --use_adj_frame_pred --model UniGRUVAE_MotionNet \
# --logpath /mnt/personal/qiankangan/logs/UniGRUVAE_aug_bs2_MotionNet --general --gpu $GPU \
# --sensor_type $sensor_type --use_dir_aug

# MotionNet + aug
# OMP_NUM_THREADS=1 python train_multi_seq.py --data /mnt/personal/qiankangan/input-data --evaldata /mnt/personal/qiankangan/val \
# --batch 8 --nepoch 45 --nworker 4 --lr 0.0016 \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 \
# --log --model En_MotionNet \
# --logpath /mnt/personal/qiankangan/logs/MotionNet_aug_bs8_MotionNet --general --gpu $GPU \
# --use_dir_aug

# MotionNet + aug
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python train_multi_seq.py --data /mnt/personal/qiankangan/input-data --evaldata /mnt/personal/qiankangan/val \
--batch 1 --nepoch 45 --nworker 4 --lr 0.0016 \
--use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 \
--log --model InsMotionNet \
--logpath /mnt/personal/qiankangan/logs/MotionNet_aug_bs8_InsMotionNet --general --gpu 1 \
--use_dir_aug
