export PYTHONPATH=/home/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH

sensor_type=$1
GPU=$2

# wt128_0 MotionNet
# OMP_NUM_THREADS=1 python train_multi_seq_wt128.py --data /mnt/personal/qiankangan/all_$sensor_type/train \
# --evaldata /mnt/personal/qiankangan/all_$sensor_type/val \
# --batch 8 --lr 0.0016 --nepoch 60 --nworker 2  --log --logpath /mnt/personal/qiankangan/logs/MotionNet/maskall_withabsvwithdir_$sensor_type --gpu $GPU \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 \
# --use_v --use_vdir --general

# wt128_0 be-sti
# OMP_NUM_THREADS=1 python train_multi_seq_wt128.py --data /mnt/personal/qiankangan/all_$sensor_type/train \
# --evaldata /mnt/personal/qiankangan/all_$sensor_type/val \
# --batch 8 --lr 0.0016 --nepoch 60 --nworker 2  --log --logpath /mnt/personal/qiankangan/logs/be-sti/maskall_withabsvwithdir_$sensor_type --gpu $GPU \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 \
# --use_v --use_vdir --backbone be-sti --general

# lower input frames
# OMP_NUM_THREADS=1 python train_multi_seq_wt128.py --data /mnt/personal/qiankangan/all_$sensor_type/train \
# --evaldata /mnt/personal/qiankangan/all_$sensor_type/val --model LowerGeneralMotionNet \
# --batch 8 --lr 0.0016 --nepoch 60 --nworker 2  --log --logpath /mnt/personal/qiankangan/logs/MotionNet/alllower_withabsvwithdir_$sensor_type --gpu $GPU \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 \
# --input_frames 1 --use_v --use_vdir


# OMP_NUM_THREADS=1 python train_multi_seq.py --data /mnt/personal/qiankangan/input-data \
# --evaldata /mnt/personal/qiankangan/val \
# --batch 8 --lr 0.0016 --nepoch 60 --nworker 2  --log --logpath /home/qiankangan/logs/MotionNet/be-sti_$sensor_type --gpu $GPU \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 --general

# /home/qiankangan/MotionNet/train_muiti_seq_wt128_mix.py
# OMP_NUM_THREADS=1 python train_multi_seq_wt128_mix.py --data /mnt/personal/qiankangan/all_$sensor_type/train \
# --evaldata /mnt/personal/qiankangan/all_$sensor_type/val \
# --batch 8 --lr 0.0016 --nepoch 60 --nworker 2  --log --logpath /mnt/personal/qiankangan/logs/MotionNet/all_mixsewithabsvwithdir_$sensor_type --gpu $GPU \
# --model MixGeneralMotionNet \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 \
# --use_v --use_vdir

# nuscenes VAE with mask
# python train_VAE.py --data /mnt/personal/qiankangan/input-data --evaldata /mnt/personal/qiankangan/val \
# --batch 8 --nepoch 60 --nworker 2 --lr 0.0016 --pre_model_path /mnt/personal/qiankangan/weak_WeakMotionNet.pth \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
# --log --auxiliary --use_adj_frame_pred --model VAE_MotionNet \
# --logpath /mnt/personal/qiankangan/logs/VAE_MotionNet_bs8_$sensor_type --general --gpu $GPU

# nuscenes VAE without mask
# python train_VAE.py --data /mnt/personal/qiankangan/input-data --evaldata /mnt/personal/qiankangan/val \
# --batch 8 --nepoch 60 --nworker 2 --lr 0.0016 --pre_model_path /mnt/personal/qiankangan/weak_WeakMotionNet.pth \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
# --log --auxiliary --use_adj_frame_pred --model VAE_MotionNet \
# --logpath /mnt/personal/qiankangan/logs/VAE_MotionNetnomask_bs8_$sensor_type --gpu $GPU

# wt128_0 be-sti VAE
python train_VAE.py --data /mnt/personal/qiankangan/all_$sensor_type/train --evaldata /mnt/personal/qiankangan/all_$sensor_type/val \
--batch 8 --nepoch 60 --nworker 2 --lr 0.0016 \
--use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
--log --auxiliary --use_adj_frame_pred --model VAE_GeneralBESTI \
--logpath /mnt/personal/qiankangan/logs/be-sti/VAE_be-sti_bs8_withnoabsv_$sensor_type --general --gpu $GPU \
--sensor_type $sensor_type
# --pre_model_path /mnt/personal/qiankangan/weak_WeakMotionNet.pth

# wt128_0 MotionNet VAE
# python train_VAE.py --data /mnt/personal/qiankangan/all_$sensor_type/train --evaldata /mnt/personal/qiankangan/all_$sensor_type/val \
# --batch 8 --nepoch 60 --nworker 2 --lr 0.0016 \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
# --log --auxiliary --use_adj_frame_pred --model VAE_GeneralMotionNet --sensor_type $sensor_type \
# --logpath /mnt/personal/qiankangan/logs/VAE_GeneralMotionNet_withabsv_bs8_$sensor_type --general --gpu $GPU \
# --use_v
