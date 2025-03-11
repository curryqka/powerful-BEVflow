#!/bin/bash # 定义id数组
RUN_ROOT="/SHARE/qiankangan/bevflow"
export PYTHONPATH=/workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion:$PYTHONPATH
export PYTHONPATH=/workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion/nuscenes-devkit/python-sdk:$PYTHONPA
cd /workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion
pip install -r requirements.txt
# model="MotionNet_CNN"
model=$1
ids=(0 3 209 239 265 273 350 447 469 812)
# 使用for循环遍历数组
for scene in "${ids[@]}"
do
    echo "current id is: $scene"
    # 在这里添加你需要执行的命令
    # python tools/plots_feat.py --data /datasets/nuscenes/ --version v1.0-trainval \
    # --net $model --nframe 100 \
    #     --modelpath $RUN_ROOT/logs/MotionNet_GRU/train_multi_seq/2025-01-24_10-22-25/save_model.pth \
    #     --savepath $RUN_ROOT/visualization/MotionNetGRU_visual/$model/scene_$scene --scene $scene
    #  python tools/plots_feat.py --data /datasets/nuscenes/ --version v1.0-trainval \
    # --net $model --nframe 100 \
    #     --modelpath $RUN_ROOT/logs/UniInsGridGRUVAE_bs4_MotionNet/train_multi_seq/2025-01-17_04-32-23/save_model.pth \
    #     --savepath $RUN_ROOT/visualization/priormotion_visual_1/$model/scene_$scene --scene $scene
    python tools/plots_feat.py --data /datasets/nuscenes/ --version v1.0-trainval \
    --net $model --nframe 100 \
        --modelpath $RUN_ROOT/logs/gridinsMotionNetSE_0.0016_8/train_multi_seq/2025-02-25_12-53-09/save_model.pth \
        --savepath $RUN_ROOT/visualization/${model}_visual_enhanced/$model/scene_$scene --scene $scene
done
