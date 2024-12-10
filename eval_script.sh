export PYTHONPATH=/home/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH

sensor_type=$1
# # distance wt128_0 noabsv
# python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/MotionNet/all_withnoabsv_wt128_0/train_multi_seq/2024-10-17_10-18-01/epoch_60.pth \
# --sensor_type $sensor_type \
# --split test --log ./logs/wt128_0_noabsv --bs 1 --net MotionNet

# # distance wt128_0 withabsv
# python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/MotionNet/all_withabsv_wt128_0/train_multi_seq/2024-10-17_10-16-39/epoch_60.pth \
# --sensor_type $sensor_type \
# --split test --log ./logs/wt128_0_withabsv --bs 1 --net MotionNet \
# --use_v

# # distance wt128_0 withabsv withdir
# python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/MotionNet/all_withabsvwithdir_wt128_0/train_multi_seq/2024-10-17_10-10-56/epoch_60.pth \
# --sensor_type $sensor_type \
# --split test --log ./logs/wt128_0_withabsvwithabsv --bs 1 --net MotionNet \
# --use_v --use_vdir

# # distance wt128_0 noabsv be-sti
# python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/be-sti/all_withnoabsv_wt128_0/train_multi_seq/2024-10-21_16-01-58/epoch_60.pth \
# --sensor_type $sensor_type \
# --split test --log ./logs/be-sti_wt128_0_noabsv --bs 1 --net be-sti

# # distance wt128_0 withabsv be-sti
# python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/be-sti/all_withabsv_wt128_0/train_multi_seq/2024-10-22_10-14-44/epoch_60.pth \
# --sensor_type $sensor_type \
# --split test --log ./logs/be-sti_wt128_0_withabsv --bs 1 --net be-sti \
# --use_v

# # distance wt128_0 withabsvwithdir be-sti
# python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/be-sti/all_withabsvwithdir_wt128_0/train_multi_seq/2024-10-22_10-26-13/epoch_60.pth \
# --sensor_type $sensor_type \
# --split test --log ./logs/be-sti_wt128_0_withabsvwithdir --bs 1 --net be-sti \
# --use_v --use_vdir

# # distance nuscenes
# python eval_general.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/nuscenes/train_multi_seq/2024-09-10_23-08-29/epoch_40.pth \
# --sensor_type $sensor_type --net MotionNet \
# --split test --log ./logs/MotionNet_$sensor_type --bs 1

# # distance wt128_0 alllower 1 withabsv withdir
# python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/MotionNet/alllower_withabsvwithdir_wt128_0/train_multi_seq/2024-10-24_08-18-07/save_model.pth \
# --sensor_type $sensor_type \
# --split test --log ./logs/MotionNetlower1_wt128_0_withabsvwithdir --bs 1 --net LowerGeneralMotionNet \
# --use_v --use_vdir

# # distance wt128_0 alllower 1 withabsv
# python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/MotionNet/alllower_withabsv_wt128_0/train_multi_seq/2024-10-23_00-53-06/save_model.pth \
# --sensor_type $sensor_type \
# --split test --log ./logs/MotionNetlower1_wt128_0_withabsv --bs 1 --net LowerGeneralMotionNet \
# --use_v

# # distance wt128_0 alllower 1 noabsv
# python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/MotionNet/alllower_withnoabsv_wt128_0/train_multi_seq/2024-10-23_01-01-25/save_model.pth \
# --sensor_type $sensor_type \
# --split test --log ./logs/MotionNetlower1_wt128_0_noabsv --bs 1 --net LowerGeneralMotionNet

# # distance wt128_0 alllower 3 withabsv withdir
# python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/MotionNet/alllower_withabsvwithdir_wt128_0/train_multi_seq/2024-10-22_12-28-32/save_model.pth \
# --sensor_type $sensor_type \
# --split test --log ./logs/MotionNetlower3_wt128_0_withabsvwithdir --bs 1 --net LowerGeneralMotionNet \
# --use_v --use_vdir

# # # distance wt128_0 alllower 3 withabsv
# python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/MotionNet/alllower_withabsv_wt128_0/train_multi_seq/2024-10-22_15-16-20/save_model.pth \
# --sensor_type $sensor_type \
# --split test --log ./logs/MotionNetlower3_wt128_0_withabsv --bs 1 --net LowerGeneralMotionNet \
# --use_v

# # distance wt128_0 alllower 3 noabsv
# python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/MotionNet/alllower_noabsv_wt128_0/train_multi_seq/2024-10-22_12-48-10/save_model.pth \
# --sensor_type $sensor_type \
# --split test --log ./logs/MotionNetlower3_wt128_0_noabsv --bs 1 --net LowerGeneralMotionNet

# ### ---------------------------------nuscenes---------------------------------------------
#  distance nuscenes
# python eval_general.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/nuscenes/train_multi_seq/2024-09-10_23-08-29/epoch_40.pth \
# --sensor_type $sensor_type --net MotionNet \
# --split test --log ./logs/MotionNet_$sensor_type --bs 1

# nuscenes UniGridGenMotionNet
# python eval_general.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/UniGridGenMotionNet_bs4_ablation/train_multi_seq/2024-11-06_01-15-39/save_model.pth \
# --sensor_type $sensor_type --net UniGridGenMotionNet \
# --split test --log ./logs/UniGridGenMotionNet_long_$sensor_type --bs 1

# python eval.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/UniGridGenMotionNet_bs4_ablation/train_multi_seq/2024-11-06_01-15-39/save_model.pth \
# --sensor_type $sensor_type --net UniGridGenMotionNet \
# --split test --log ./logs/UniGridGenMotionNet_$sensor_type --bs 1

# python eval.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/UniGridGenMotionNet_bs4_ablation/train_multi_seq/2024-11-06_01-15-39/save_model.pth \
# --sensor_type $sensor_type --net UniGridGenMotionNet \
# --split test --log ./logs/UniGridGenMotionNet_mask_$sensor_type --bs 1 --general

# # nuscenes /mnt/personal/qiankangan/logs/UniInsGridGRUVAE_bs4_MotionNet/train_multi_seq/2024-11-05_19-38-09/save_model.pth
# python eval_general.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/UniInsGridGRUVAE_bs4_MotionNet/train_multi_seq/2024-11-05_19-38-09/save_model.pth \
# --sensor_type $sensor_type --net UniInsGridGenMotionNet \
# --split test --log ./logs/UniInsGridGenMotionNet_long_$sensor_type --bs 1

# python eval.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/UniInsGridGRUVAE_bs4_MotionNet/train_multi_seq/2024-11-05_19-38-09/save_model.pth \
# --sensor_type $sensor_type --net UniInsGridGenMotionNet \
# --split test --log ./logs/UniInsGridGenMotionNet_$sensor_type --bs 1

# python eval.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/UniInsGridGRUVAE_bs4_MotionNet/train_multi_seq/2024-11-05_19-38-09/save_model.pth \
# --sensor_type $sensor_type --net UniInsGridGenMotionNet \
# --split test --log ./logs/UniInsGridGenMotionNet_mask_$sensor_type --bs 1 --general

# # nuscenes /mnt/personal/qiankangan/logs/MulUniGridInsGRUMotionNetVAE64x64_bs4_ablation/train_multi_seq/2024-11-05_15-36-24/save_model.pth
# python eval_general.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/MulUniGridInsGRUMotionNetVAE64x64_bs4_ablation/train_multi_seq/2024-11-05_15-36-24/save_model.pth \
# --sensor_type $sensor_type --net MulUniGridInsGRUMotionNetVAE64x64 \
# --split test --log ./logs/MulUniGridInsGRUMotionNetVAE64x64_long_$sensor_type --bs 1

# python eval.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/MulUniGridInsGRUMotionNetVAE64x64_bs4_ablation/train_multi_seq/2024-11-05_15-36-24/save_model.pth \
# --sensor_type $sensor_type --net MulUniGridInsGRUMotionNetVAE64x64 \
# --split test --log ./logs/MulUniGridInsGRUMotionNetVAE64x64_$sensor_type --bs 1

# python eval.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/MulUniGridInsGRUMotionNetVAE64x64_bs4_ablation/train_multi_seq/2024-11-05_15-36-24/save_model.pth \
# --sensor_type $sensor_type --net MulUniGridInsGRUMotionNetVAE64x64 \
# --split test --log ./logs/MulUniGridInsGRUMotionNetVAE64x64_mask_$sensor_type --bs 1 --general

# # nuscenes /mnt/personal/qiankangan/logs/MulUniGridInsGRUMotionNetVAE32x32_bs4_ablation/train_multi_seq/2024-11-05_15-34-38/save_model.pth
# python eval_general.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/MulUniGridInsGRUMotionNetVAE32x32_bs4_ablation/train_multi_seq/2024-11-05_15-34-38/save_model.pth \
# --sensor_type $sensor_type --net MulUniGridInsGRUMotionNetVAE32x32 \
# --split test --log ./logs/MulUniGridInsGRUMotionNetVAE32x32_long_$sensor_type --bs 1

# python eval.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/MulUniGridInsGRUMotionNetVAE32x32_bs4_ablation/train_multi_seq/2024-11-05_15-34-38/save_model.pth \
# --sensor_type $sensor_type --net MulUniGridInsGRUMotionNetVAE32x32 \
# --split test --log ./logs/MulUniGridInsGRUMotionNetVAE32x32_$sensor_type --bs 1

# python eval.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/MulUniGridInsGRUMotionNetVAE32x32_bs4_ablation/train_multi_seq/2024-11-05_15-34-38/save_model.pth \
# --sensor_type $sensor_type --net MulUniGridInsGRUMotionNetVAE32x32 \
# --split test --log ./logs/MulUniGridInsGRUMotionNetVAE32x32_mask_$sensor_type --bs 1 --general