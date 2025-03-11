export PYTHONPATH=/workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion:$PYTHONPATH
export PYTHONPATH=/workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion/nuscenes-devkit/python-sdk:$PYTHONPA
# /SHARE/qiankangan/bevflow/visualization/MotionNet_visual/MotionNet/scene_3
RUN_ROOT="/SHARE/qiankangan/bevflow"
scene=$1
# ---------input frame1----------
python tools/plots_feat.py --data /datasets/nuscenes/ --version v1.0-trainval \
--modelpath $RUN_ROOT/logs/MotionNet/train_multi_seq/2025-02-14_08-48-16/save_model.pth \
--framepath $RUN_ROOT/visualization/priormotion_visual_1/UniGridInsGRUMotionNetVAE/scene_${scene} \
    --net UniGridInsGRUMotionNetVAE --nframe 100 --savepath $RUN_ROOT/visualization_video_${scene}  --scene $scene \
    --beginframe 1 --video
