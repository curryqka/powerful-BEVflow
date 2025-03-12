model=$1
# scene=$2
RUN_ROOT="/SHARE/qiankangan/bevflow"
export PYTHONPATH=/workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion:$PYTHONPATH
export PYTHONPATH=/workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion/nuscenes-devkit/python-sdk:$PYTHONPA
cd /workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion
pip install -r requirements.txt
ids=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 209 239 265 273 350 447 469 812)
# 使用for循环遍历数组
for scene in "${ids[@]}"
do
    echo "current id is: $scene"
    python tools/plots_feat.py --data /datasets/nuscenes/ --version v1.0-trainval \
    --net $model --nframe 300 \
        --modelpath $RUN_ROOT/logs/UniInsGridGRUVAE_bs4_MotionNet/train_multi_seq/2025-01-17_04-32-23/save_model.pth \
        --savepath $RUN_ROOT/visualization/priormotion_visual/$model/scene_$scene --scene $scene
done

# # UniGridInsGRUMotionNetVAE
# python tools/plots_feat.py --data /datasets/nuscenes/ --version v1.0-trainval \
# --net $model --nframe 300 \
#     --modelpath $RUN_ROOT/logs/UniInsGridGRUVAE_bs4_MotionNet/train_multi_seq/2025-01-17_04-32-23/save_model.pth \
#     --savepath $RUN_ROOT/visualization/priormotion_visual/$model/scene_$scene --scene $scene

# MotionNet
# python tools/plots_feat.py --data /datasets/nuscenes/ --version v1.0-trainval \
# --net $model --nframe 100 \
#     --modelpath $RUN_ROOT/logs/MotionNet/train_multi_seq/2025-01-17_04-16-51/save_model.pth \
#     --savepath $RUN_ROOT/visualization/MotionNet_visual/$model/scene_$scene --scene $scene

# MotionNet_GRU
# python tools/plots_feat.py --data /datasets/nuscenes/ --version v1.0-trainval \
# --net $model --nframe 100 \
#     --modelpath $RUN_ROOT/logs/MotionNet_GRU/train_multi_seq/2025-01-24_10-22-25/save_model.pth \
#     --savepath $RUN_ROOT/visualization/MotionNetGRU_visual/$model/scene_$scene --scene $scene

# MotionNet_CNN
# python tools/plots_feat.py --data /datasets/nuscenes/ --version v1.0-trainval \
# --net $model --nframe 100 \
#     --modelpath $RUN_ROOT/logs/MotionNet_CNN/train_multi_seq/2025-01-24_09-59-24/save_model.pth \
#     --savepath $RUN_ROOT/visualization/MotionNetCNN_visual/$model/scene_$scene --scene $scene

# # MotionNetHigher
# python tools/plots_feat.py --data /datasets/nuscenes/ --version v1.0-trainval \
# --net $model --nframe 100 \
#     --modelpath $RUN_ROOT/logs/MotionNetHigher/train_multi_seq/2025-01-25_03-48-22/save_model.pth \
#     --savepath $RUN_ROOT/visualization/MotionNetHigher_visual/$model/scene_$scene --scene $scene
