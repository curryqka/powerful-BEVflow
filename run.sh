export PYTHONPATH=/home/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH

sensor_type=$1
GPU=$2

OMP_NUM_THREADS=1 python train_multi_seq_wt128.py --data /mnt/personal/qiankangan/V0_$sensor_type/train \
--evaldata /mnt/personal/qiankangan/V0_$sensor_type/val \
--batch 32 --lr 0.0032 --nepoch 80 --nworker 2  --log --logpath /home/qiankangan/logs/V0_noabsv_$sensor_type --gpu $GPU \
--use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 \
# --use_v

# OMP_NUM_THREADS=1 python train_multi_seq.py --data /mnt/personal/qiankangan/$sensor_type/train \
# --evaldata /mnt/personal/qiankangan/$sensor_type/val \
# --batch 16 --nepoch 80 --nworker 2  --log --logpath /home/qiankangan/logs/$sensor_type --gpu $GPU \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 