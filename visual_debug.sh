export PYTHONPATH=/home/hesai/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/hesai/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH
export PYTHONPATH=/mnt/personal/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/mnt/personal/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH
# coding=utf-8
export PYTHONPATH=/home/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH

# python visual_debug.py --data /mnt/yrfs/Team/Semantic/datasets/AllData/HaData  --modelpath  \
# /mnt/personal/qiankangan/logs/at128_0/train_multi_seq/2024-09-04_11-40-09/epoch_78.pth \
#     --net MotionNet --nframe 100 --savepath ./output --scene 0 --adj --datavision V1

python visual_debug.py --data /mnt/personal/qiankangan/hadata  --modelpath  \
/mnt/personal/qiankangan/logs/at128_0/train_multi_seq/2024-09-04_11-40-09/epoch_78.pth \
    --net MotionNet --nframe 100 --savepath ./output --scene 25 --adj --datavision V1

# python plot_res_point.py --data /mnt/yrfs/Team/Semantic/datasets/AllData/HaData  --modelpath  \
# /mnt/personal/qiankangan/logs/at128_0/train_multi_seq/2024-09-04_11-40-09/epoch_78.pth \
#     --net MotionNet --nframe 100 --savepath ./results --scene 10 --adj --datavision V1
