export PYTHONPATH=/home/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH

sensor_type=$1
GPU=$2

OMP_NUM_THREADS=1 python train_multi_seq_wt128.py --data /mnt/personal/qiankangan/all_$sensor_type/train \
--evaldata /mnt/personal/qiankangan/V0new_$sensor_type/val \
--batch 8 --lr 0.0016 --nepoch 100 --nworker 2  --log --logpath /mnt/personal/qiankangan/logs/MotionNet/all_withabsv_$sensor_type --gpu $GPU \
--use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 \
--use_v

# OMP_NUM_THREADS=1 python train_multi_seq.py --data /mnt/personal/qiankangan/$sensor_type/train \
# --evaldata /mnt/personal/qiankangan/$sensor_type/val \
# --batch 8 --lr 0.0016 --nepoch 50 --nworker 2  --log --logpath /home/qiankangan/logs/MotionNet/$sensor_type --gpu $GPU \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 

# /home/qiankangan/MotionNet/train_muiti_seq_wt128_mix.py
# OMP_NUM_THREADS=1 python train_multi_seq_wt128_mix.py --data /mnt/personal/qiankangan/V3_$sensor_type/train \
# --evaldata /mnt/personal/qiankangan/V3_$sensor_type/val \
# --batch 8 --lr 0.0016 --nepoch 100 --nworker 2  --log --logpath /mnt/personal/qiankangan/logs/MotionNet/V3_noabsv_$sensor_type --gpu $GPU \
# --model MixGeneralMotionNet \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 \
# --use_v