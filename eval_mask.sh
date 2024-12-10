export PYTHONPATH=/home/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH

sensor_type=$1
gpu=$2
# distance wt128_0 noabsv
# python eval.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/VAE_GeneralMotionNet_bs16_wt128_0/train_multi_seq/2024-10-19_18-02-40/save_model.pth \
# --sensor_type $sensor_type  --net UniGenMotionNet \
# --split test --log ./logs/wt128_0_gen_noabsv_mask --bs 1 \
# --general --gpu $gpu

# # distance wt128_0 withabsv
# # python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# # --model /mnt/personal/qiankangan/logs/MotionNet/all_withabsv_wt128_0/train_multi_seq/2024-10-17_10-16-39/epoch_60.pth \
# # --sensor_type $sensor_type --net UniGenMotionNet \
# # --split test --log ./logs/wt128_0_gen_withabsv --bs 1 \
# # --use_v

# # distance wt128_0 withabsv withdir
# python eval.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/VAE_GeneralMotionNet_bs16_withabsvwithdir_wt128_0/train_multi_seq/2024-10-20_07-59-23/save_model.pth \
# --sensor_type $sensor_type --net UniGenMotionNet \
# --split test --log ./logs/wt128_0_gen_withabsvwithabsv_mask --bs 1 \
# --use_v --use_vdir --general --gpu $gpu

# # distance wt128_0 noabsv
# python eval.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/MotionNet/all_withnoabsv_wt128_0/train_multi_seq/2024-10-17_10-18-01/epoch_60.pth \
# --sensor_type $sensor_type \
# --split test --log ./logs/wt128_0_noabsv_mask --bs 1 --net MotionNet \
# --general --gpu $gpu

# # distance wt128_0 withabsv
# python eval.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/MotionNet/all_withabsv_wt128_0/train_multi_seq/2024-10-17_10-16-39/epoch_60.pth \
# --sensor_type $sensor_type \
# --split test --log ./logs/wt128_0_withabsv_mask --bs 1 --net MotionNet \
# --use_v --general --gpu $gpu

# # distance wt128_0 withabsv withdir
# python eval.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/MotionNet/all_withabsvwithdir_wt128_0/train_multi_seq/2024-10-17_10-10-56/epoch_60.pth \
# --sensor_type $sensor_type \
# --split test --log ./logs/wt128_0_withabsvwithabsv_mask --bs 1 --net MotionNet \
# --use_v --use_vdir --general --gpu $gpu

# nuscenes unigen mask
# python eval.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/VAE_MotionNet_bs8_nuscenes/train_multi_seq/2024-10-19_01-02-22/save_model.pth \
# --sensor_type nuscenes --net UniGenMotionNet \
# --split test --log ./logs/unigen_nuscenes_mask --bs 1 --general

# ----------------------------------------------------------------------------
# for BESTI
# wt128_0 besti noabsv mask
python eval.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
--model /mnt/personal/qiankangan/logs/be-sti/all_withnoabsv_wt128_0/train_multi_seq/2024-10-21_16-01-58/save_model.pth \
--sensor_type $sensor_type \
--split test --log ./logs/be-sti_wt128_0_noabsv_mask --bs 1 --net be-sti \
--general --gpu $gpu

# wt128_0 besti withabsv mask
python eval.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
--model /mnt/personal/qiankangan/logs/be-sti/all_withabsv_wt128_0/train_multi_seq/2024-10-22_10-14-44/save_model.pth \
--sensor_type $sensor_type \
--split test --log ./logs/be-sti_wt128_0_withabsv_mask --bs 1 --net be-sti \
--use_v --general --gpu $gpu

# wt128_0 besti withabsv withdir mask
python eval.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
--model /mnt/personal/qiankangan/logs/be-sti/all_withabsvwithdir_wt128_0/train_multi_seq/2024-10-22_10-26-13/save_model.pth \
--sensor_type $sensor_type \
--split test --log ./logs/be-sti_wt128_0_withabsvwithabsv_mask --bs 1 --net be-sti \
--use_v --use_vdir --general --gpu $gpu


