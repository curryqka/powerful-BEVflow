export PYTHONPATH=/home/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH
python plotsH.py --data /mnt/yrfs/Team/Semantic/datasets/AllData/HaData  --modelpath logs/train_multi_seq/2024-08-18_18-29-38/epoch_80.pth \
    --net MotionNet --nframe 100 --savepath ./results --scene 25