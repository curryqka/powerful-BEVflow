export PYTHONPATH=/home/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH

sensor_type=$1
# # distance wt128_0 noabsv
# python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/VAE_GeneralMotionNet_bs16_wt128_0/train_multi_seq/2024-10-19_18-02-40/save_model.pth \
# --sensor_type $sensor_type  --net UniGenMotionNet \
# --split test --log ./logs/wt128_0_gen_noabsv --bs 1

# # distance wt128_0 withabsv
# # python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# # --model /mnt/personal/qiankangan/logs/MotionNet/all_withabsv_wt128_0/train_multi_seq/2024-10-17_10-16-39/epoch_60.pth \
# # --sensor_type $sensor_type --net UniGenMotionNet \
# # --split test --log ./logs/wt128_0_gen_withabsv --bs 1 \
# # --use_v

# # distance wt128_0 withabsv withdir
# python eval_general.py --data /mnt/personal/qiankangan/all_$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/VAE_GeneralMotionNet_bs16_withabsvwithdir_wt128_0/train_multi_seq/2024-10-20_07-59-23/save_model.pth \
# --sensor_type $sensor_type --net UniGenMotionNet \
# --split test --log ./logs/wt128_0_gen_withabsvwithabsv --bs 1 \
# --use_v --use_vdir

# distance nuscenes MotionNet
python eval_general.py --data /mnt/personal/qiankangan/test \
--model /mnt/personal/qiankangan/logs/nuscenes/train_multi_seq/2024-09-10_23-08-29/epoch_40.pth \
--sensor_type $sensor_type --net MotionNet \
--split test --log ./logs/MotionNet_$sensor_type --bs 1

# # distance nuscenes GenMotionNet
python eval_general.py --data /mnt/personal/qiankangan/test \
--model /mnt/personal/qiankangan/logs/VAE_MotionNet_bs8_nuscenes/train_multi_seq/2024-10-19_01-02-22/save_model.pth \
--sensor_type $sensor_type --net UniGenMotionNet \
--split test --log ./logs/unigen_$sensor_type --bs 1


