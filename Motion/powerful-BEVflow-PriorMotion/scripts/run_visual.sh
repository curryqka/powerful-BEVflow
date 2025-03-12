#!/bin/bash # 定义id数组
RUN_ROOT="/SHARE/qiankangan/bevflow"
export PYTHONPATH=/workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion:$PYTHONPATH
export PYTHONPATH=/workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion/nuscenes-devkit/python-sdk:$PYTHONPA
cd /workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion
pip install -r requirements.txt
# model="MotionNet_CNN"
model=$1
ids=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 209 239 265 273 350 447 469 812)
# 使用for循环遍历数组
for scene in "${ids[@]}"
do
    echo "current id is: $scene"
    python tools/plots_feat.py --data /datasets/nuscenes/ --version v1.0-trainval \
    --net $model --nframe 300 \
        --modelpath $RUN_ROOT/logs/gridinsMotionNetSE_0.0016_8/train_multi_seq/2025-02-25_12-53-09/save_model.pth \
        --savepath $RUN_ROOT/visualization/${model}_visual_noise/$model/scene_$scene --scene $scene
done
