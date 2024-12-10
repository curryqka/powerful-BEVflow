export PYTHONPATH=/home/qiankangan/UniGenMotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/UniGenMotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH

python /home/qiankangan/UniGenMotionNet/data/read_split.py
scenes=$(cat /home/qiankangan/UniGenMotionNet/data/visual_val_list.txt)
# -----------------------nuscenes------------------------------
# visual motionnet nuscenes
# /mnt/personal/qiankangan/logs/nuscenes/train_multi_seq/2024-09-10_23-08-29/epoch_45.pth
model=$1
for a in $scenes
do
    echo $a
    python plots.py --data /mnt/yrfs/Team/Semantic/datasets/AllData/nuScenes/raw_data/v1.0-trainval/ --version v1.0-trainval --net $model --nframe 100 \
    --modelpath /mnt/personal/qiankangan/logs/GRUVAE_MotionNet/train_multi_seq/2024-10-25_23-06-12/save_model.pth \
    --savepath /mnt/personal/qiankangan/priormotion_visual/$model/scene_$a --scene $a
done

# for a in {600..620}
# do
#     echo $a
#     python plots.py --data /mnt/yrfs/Team/Semantic/datasets/AllData/nuScenes/raw_data/v1.0-trainval/ --version v1.0-trainval --net $model --nframe 100 \
#     --modelpath /mnt/personal/qiankangan/logs/GRUVAE_MotionNet/train_multi_seq/2024-10-25_23-06-12/save_model.pth \
#     --savepath /mnt/personal/qiankangan/priormotion_visual/$model/scene_$a --scene $a
# done


# python plots.py --data /mnt/yrfs/Team/Semantic/datasets/AllData/nuScenes/raw_data/v1.0-trainval/ --version v1.0-trainval --net $model --nframe 100 \
#     --modelpath /mnt/personal/qiankangan/logs/nuscenes/train_multi_seq/2024-09-10_23-08-29/epoch_40.pth \
#     --savepath /mnt/personal/qiankangan/priormotion_visual/$model/scene_$scene --scene $scene

# python plots.py --data /mnt/yrfs/Team/Semantic/datasets/AllData/nuScenes/raw_data/v1.0-trainval/ --version v1.0-trainval --net $model --nframe 100 \
#     --modelpath /mnt/personal/qiankangan/logs/GRUVAE_MotionNet/train_multi_seq/2024-10-25_23-06-12/save_model.pth \
#     --savepath /mnt/personal/qiankangan/priormotion_visual/$model/scene_$scene --scene $scene

