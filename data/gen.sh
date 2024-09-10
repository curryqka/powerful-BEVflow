# coding=utf-8
export PYTHONPATH=/home/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH

data_type=$1
lidar_type=$2
vision=$3
echo "save path is : /mnt/personal/qiankangan/$lidar_type/$data_type"
# motionnet
OMP_NUM_THREADS=1 python data/gen_Hdata.py  --root /mnt/yrfs/Team/Semantic/datasets/AllData/HaData2 \
--split $data_type --lidartype $lidar_type --savepath /mnt/personal/qiankangan/V2_$lidar_type/$data_type \
--datavision $vision