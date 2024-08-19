# coding=utf-8
export PYTHONPATH=/home/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH

# motionnet
OMP_NUM_THREADS=1 python data/gen_Hdata.py  --root /mnt/yrfs/Team/Semantic/datasets/AllData/HaData \
--split train --savepath /mnt/personal/qiankangan/at128_0/train