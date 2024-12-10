export PYTHONPATH=/home/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH

scene=$1
# ---------input frame1----------
python plotsH.py --data /mnt/personal/qiankangan/hadata --version v1.0-trainval \
--modelpath /mnt/personal/qiankangan/logs/MotionNet/alllower_withnoabsv_wt128_0/train_multi_seq/2024-10-23_01-01-25/save_model.pth \
    --net LowerGeneralMotionNet --nframe 100 --savepath ./results/alllower_withnoabsv_wt128_0  --scene $scene \
    --sensor_type wt128_0 --input_frames 1 --video

# visual lower motionnet withabsv
python plotsH.py --data /mnt/personal/qiankangan/hadata --version v1.0-trainval \
--modelpath /mnt/personal/qiankangan/logs/MotionNet/alllower_withabsv_wt128_0/train_multi_seq/2024-10-23_00-53-06/save_model.pth \
    --net LowerGeneralMotionNet --nframe 100 --savepath ./results/alllower_withabsv_wt128_0 --scene $scene \
    --sensor_type wt128_0 --use_v --input_frames 1 --video

# visual lower motionnet withabsvwithdir
python plotsH.py --data /mnt/personal/qiankangan/hadata --version v1.0-trainval \
--modelpath /mnt/personal/qiankangan/logs/MotionNet/alllower_withabsvwithdir_wt128_0/train_multi_seq/2024-10-24_07-39-32/save_model.pth \
    --net LowerGeneralMotionNet --nframe 100 --savepath ./results/alllower_withabsvwithdir_wt128_0 --scene $scene \
    --sensor_type wt128_0 --use_v --use_vdir --input_frames 1 --video

# ---------input frame3----------
# visual lower motionnet3 noabsv
python plotsH.py --data /mnt/personal/qiankangan/hadata --version v1.0-trainval \
--modelpath /mnt/personal/qiankangan/logs/MotionNet/alllower_noabsv_wt128_0/train_multi_seq/2024-10-22_12-48-10/save_model.pth \
    --net LowerGeneralMotionNet --nframe 100 --savepath ./results/alllower3_withnoabsv_wt128_0 --scene $scene \
    --sensor_type wt128_0 --input_frames 3 --video

# visual lower motionnet3 withabsv
python plotsH.py --data /mnt/personal/qiankangan/hadata --version v1.0-trainval \
--modelpath /mnt/personal/qiankangan/logs/MotionNet/alllower_withabsv_wt128_0/train_multi_seq/2024-10-22_15-16-20/save_model.pth \
    --net LowerGeneralMotionNet --nframe 100 --savepath ./results/alllower3_withabsv_wt128_0 --scene $scene \
    --sensor_type wt128_0 --use_v --input_frames 3 --video

# visual lower motionnet3 withabsv withdir
python plotsH.py --data /mnt/personal/qiankangan/hadata --version v1.0-trainval \
--modelpath /mnt/personal/qiankangan/logs/MotionNet/alllower_withabsvwithdir_wt128_0/train_multi_seq/2024-10-22_12-28-32/save_model.pth \
    --net LowerGeneralMotionNet --nframe 100 --savepath ./results/alllower3_withabsvwithdir_wt128_0 --scene $scene \
    --sensor_type wt128_0 --use_v --use_vdir --input_frames 3 --video


# visual motionnet noabsv
python plotsH.py --data /mnt/personal/qiankangan/hadata --version v1.0-trainval \
--modelpath /mnt/personal/qiankangan/logs/MotionNet/all_withnoabsv_wt128_0/train_multi_seq/2024-10-17_10-18-01/epoch_60.pth \
    --net MotionNet --nframe 100 --savepath ./results/all_withnoabsv_wt128_0 --scene $scene \
    --sensor_type wt128_0 --video

# visual motionnet withabsv
python plotsH.py --data /mnt/personal/qiankangan/hadata --version v1.0-trainval \
--modelpath /mnt/personal/qiankangan/logs/MotionNet/all_withabsv_wt128_0/train_multi_seq/2024-10-17_10-16-39/epoch_60.pth \
    --net MotionNet --nframe 100 --savepath ./results/all_withabsv_wt128_0 --scene $scene \
    --sensor_type wt128_0 --use_v --video

# visual motionnet withabsv withdir
python plotsH.py --data /mnt/personal/qiankangan/hadata --version v1.0-trainval \
--modelpath /mnt/personal/qiankangan/logs/MotionNet/all_withabsvwithdir_wt128_0/train_multi_seq/2024-10-17_10-10-56/epoch_60.pth \
    --net MotionNet --nframe 100 --savepath ./results/all_withabsvwithdir_wt128_0 --scene $scene \
    --sensor_type wt128_0 --use_v --use_vdir --video

# visual motionnet noabsv
python plotsH.py --data /mnt/personal/qiankangan/hadata --version v1.0-trainval \
--modelpath /mnt/personal/qiankangan/logs/VAE_GeneralMotionNet_bs16_wt128_0/train_multi_seq/2024-10-19_18-02-40/save_model.pth \
    --net UniGenMotionNet --nframe 100 --savepath ./results/allgen_noabsv_wt128_0 --scene $scene \
    --sensor_type wt128_0 --video

# visual motionnet withabsv
python plotsH.py --data /mnt/personal/qiankangan/hadata --version v1.0-trainval \
--modelpath /mnt/personal/qiankangan/logs/VAE_GeneralMotionNet_withabsv_bs8_wt128_0/train_multi_seq/2024-10-21_17-45-55/save_model.pth \
    --net UniGenMotionNet --nframe 100 --savepath ./results/allgen_withabsv_wt128_0 --scene $scene \
    --sensor_type wt128_0 --use_v --video

# visual motionnet withabsv withdir
python plotsH.py --data /mnt/personal/qiankangan/hadata --version v1.0-trainval \
--modelpath /mnt/personal/qiankangan/logs/VAE_GeneralMotionNet_bs16_withabsvwithdir_wt128_0/train_multi_seq/2024-10-20_07-59-23/save_model.pth \
    --net UniGenMotionNet --nframe 100 --savepath ./results/allgen_withabsvwithdir_wt128_0 --scene $scene \
    --sensor_type wt128_0 --use_v --use_vdir --video