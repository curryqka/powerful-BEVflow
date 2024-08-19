# coding=utf-8
export PYTHONPATH=/home/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH

# motionnet
python data/gen_data.py --root /mnt/yrfs/Team/Semantic/datasets/AllData/nuScenes/raw_data/v1.0-trainval/ --split test \
        --savepath /mnt/yrfs/Personal/qiankangan2/test

# python data/gen_data.py --root /mnt/yrfs/Team/Semantic/datasets/AllData/nuScenes/raw_data/v1.0-trainval/ --split train \
#         --savepath /mnt/yrfs/Personal/TsinghuaShare/qiankangan