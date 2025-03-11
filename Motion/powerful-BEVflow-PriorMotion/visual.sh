export PYTHONPATH=/home/qiankangan/UniGenMotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/UniGenMotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH

# scene=$1
# # visual lower motionnet withnoabsv
# python plotsH.py --data /mnt/personal/qiankangan/hadata \
# --modelpath /mnt/personal/qiankangan/logs/MotionNet/alllower_withnoabsv_wt128_0/train_multi_seq/2024-10-23_01-01-25/save_model.pth \
#     --net LowerGeneralMotionNet --nframe 100 --savepath ./results/alllower_withnoabsv_wt128_0 --scene $scene \
#     --sensor_type wt128_0 --input_frames 1

# # visual lower motionnet withabsv
# python plotsH.py --data /mnt/personal/qiankangan/hadata \
# --modelpath /mnt/personal/qiankangan/logs/MotionNet/alllower_withabsv_wt128_0/train_multi_seq/2024-10-23_00-53-06/save_model.pth \
#     --net LowerGeneralMotionNet --nframe 100 --savepath ./results/alllower_withabsv_wt128_0 --scene $scene \
#     --sensor_type wt128_0 --use_v --input_frames 1

# # visual lower motionnet withabsvwithdir
# python plotsH.py --data /mnt/personal/qiankangan/hadata \
# --modelpath /mnt/personal/qiankangan/logs/MotionNet/alllower_withabsvwithdir_wt128_0/train_multi_seq/2024-10-24_07-39-32/save_model.pth \
#     --net LowerGeneralMotionNet --nframe 100 --savepath ./results/alllower_withabsvwithdir_wt128_0 --scene $scene \
#     --sensor_type wt128_0 --use_v --use_vdir --input_frames 1

# ---------input frame----------
# visual lower motionnet3 noabsv
# python plotsH.py --data /mnt/personal/qiankangan/hadata \
# --modelpath /mnt/personal/qiankangan/logs/MotionNet/alllower_noabsv_wt128_0/train_multi_seq/2024-10-22_12-48-10/save_model.pth \
#     --net LowerGeneralMotionNet --nframe 100 --savepath ./results/alllower3_withnoabsv_wt128_0 --scene $scene \
#     --sensor_type wt128_0 --input_frames 3

# # visual lower motionnet3 withabsv
# python plotsH.py --data /mnt/personal/qiankangan/hadata \
# --modelpath /mnt/personal/qiankangan/logs/MotionNet/alllower_withabsv_wt128_0/train_multi_seq/2024-10-22_15-16-20/save_model.pth \
#     --net LowerGeneralMotionNet --nframe 100 --savepath ./results/alllower3_withabsv_wt128_0 --scene $scene \
#     --sensor_type wt128_0 --use_v --input_frames 3

# # visual lower motionnet3 withabsv withdir
# python plotsH.py --data /mnt/personal/qiankangan/hadata \
# --modelpath /mnt/personal/qiankangan/logs/MotionNet/alllower_withabsvwithdir_wt128_0/train_multi_seq/2024-10-22_12-28-32/save_model.pth \
#     --net LowerGeneralMotionNet --nframe 100 --savepath ./results/alllower3_withabsvwithdir_wt128_0 --scene $scene \
#     --sensor_type wt128_0 --use_v --use_vdir --input_frames 3

# # visual motionnet noabsv
# python plotsH.py --data /mnt/personal/qiankangan/hadata \
# --modelpath /mnt/personal/qiankangan/logs/MotionNet/all_withnoabsv_wt128_0/train_multi_seq/2024-10-17_10-18-01/epoch_60.pth \
#     --net MotionNet --nframe 100 --savepath ./results/all_withnoabsv_wt128_0 --scene $scene \
#     --sensor_type wt128_0

# # visual motionnet withabsv
# python plotsH.py --data /mnt/personal/qiankangan/hadata \
# --modelpath /mnt/personal/qiankangan/logs/MotionNet/all_withabsv_wt128_0/train_multi_seq/2024-10-17_10-16-39/epoch_60.pth \
#     --net MotionNet --nframe 100 --savepath ./results/all_withabsv_wt128_0 --scene $scene \
#     --sensor_type wt128_0 --use_v

# # visual motionnet withabsv withdir
# python plotsH.py --data /mnt/personal/qiankangan/hadata \
# --modelpath /mnt/personal/qiankangan/logs/MotionNet/all_withabsvwithdir_wt128_0/train_multi_seq/2024-10-17_10-10-56/epoch_60.pth \
#     --net MotionNet --nframe 100 --savepath ./results/all_withabsvwithdir_wt128_0 --scene $scene \
#     --sensor_type wt128_0 --use_v --use_vdir

# # visual motionnet noabsv
# python plotsH.py --data /mnt/personal/qiankangan/hadata \
# --modelpath /mnt/personal/qiankangan/logs/VAE_GeneralMotionNet_bs16_wt128_0/train_multi_seq/2024-10-19_18-02-40/save_model.pth \
#     --net UniGenMotionNet --nframe 100 --savepath ./results/allgen_noabsv_wt128_0 --scene $scene \
#     --sensor_type wt128_0

# # visual motionnet withabsv
# python plotsH.py --data /mnt/personal/qiankangan/hadata \
# --modelpath /mnt/personal/qiankangan/logs/VAE_GeneralMotionNet_withabsv_bs8_wt128_0/train_multi_seq/2024-10-21_17-45-55/save_model.pth \
#     --net UniGenMotionNet --nframe 100 --savepath ./results/allgen_withabsv_wt128_0 --scene $scene \
#     --sensor_type wt128_0 --use_v

# # visual motionnet withabsv withdir
# python plotsH.py --data /mnt/personal/qiankangan/hadata \
# --modelpath /mnt/personal/qiankangan/logs/VAE_GeneralMotionNet_bs16_withabsvwithdir_wt128_0/train_multi_seq/2024-10-20_07-59-23/save_model.pth \
#     --net UniGenMotionNet --nframe 100 --savepath ./results/allgen_withabsvwithdir_wt128_0 --scene $scene \
#     --sensor_type wt128_0 --use_v --use_vdir

# -----------------------nuscenes------------------------------
# visual motionnet nuscenes
# /mnt/personal/qiankangan/logs/nuscenes/train_multi_seq/2024-09-10_23-08-29/epoch_45.pth
model=$1
# for a in {500..510}
# do
#     echo $a
#     python plots.py --data /mnt/yrfs/Team/Semantic/datasets/AllData/nuScenes/raw_data/v1.0-trainval/ --version v1.0-trainval --net $model --nframe 100 \
#     --modelpath /mnt/personal/qiankangan/logs/nuscenes/train_multi_seq/2024-09-10_23-08-29/epoch_45.pth \
#     --savepath /mnt/personal/qiankangan/priormotion_visual/$model/scene_$a --scene $a
# done
python /home/qiankangan/UniGenMotionNet/data/read_split.py
scenes=$(cat /home/qiankangan/UniGenMotionNet/data/visual_val_list.txt)
for a in $scenes
do
    echo $a
    python plots.py --data /mnt/yrfs/Team/Semantic/datasets/AllData/nuScenes/raw_data/v1.0-trainval/ --version v1.0-trainval --net $model --nframe 100 \
    --modelpath /mnt/personal/qiankangan/logs/nuscenes/train_multi_seq/2024-09-10_23-08-29/epoch_40.pth \
    --savepath /mnt/personal/qiankangan/priormotion_visual/$model/scene_$a --scene $a
done

# python plots.py --data /mnt/yrfs/Team/Semantic/datasets/AllData/nuScenes/raw_data/v1.0-trainval/ --version v1.0-trainval --net $model --nframe 100 \
#     --modelpath /mnt/personal/qiankangan/logs/nuscenes/train_multi_seq/2024-09-10_23-08-29/epoch_40.pth \
#     --savepath /mnt/personal/qiankangan/priormotion_visual/$model/scene_$scene --scene $scene

# python plots.py --data /mnt/yrfs/Team/Semantic/datasets/AllData/nuScenes/raw_data/v1.0-trainval/ --version v1.0-trainval --net $model --nframe 100 \
#     --modelpath /mnt/personal/qiankangan/logs/GRUVAE_MotionNet/train_multi_seq/2024-10-25_23-06-12/save_model.pth \
#     --savepath /mnt/personal/qiankangan/priormotion_visual/$model/scene_$scene --scene $scene

