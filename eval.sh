export PYTHONPATH=/home/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH
sensor_type=$1
GPU=$2
# --model /mnt/personal/qiankangan/logs/MotionNet/all_withabsvwithdir_wt128_0/train_multi_seq/2024-10-11_08-36-09/epoch_40.pth \
# --model /mnt/personal/qiankangan/logs/MotionNet/V0_withabsvwithdir_wt128_0/train_multi_seq/2024-09-19_09-48-58/epoch_100.pth \
# --model /mnt/personal/qiankangan/logs/MotionNet/V2_withabsvwithdir_wt128_0/train_multi_seq/2024-09-18_14-23-27/epoch_100.pth \
# --model /mnt/personal/qiankangan/logs/MotionNet/all_withnoabsv_wt128_0/train_multi_seq/2024-10-11_08-35-11/epoch_40.pth \
# wt128_0
# python eval_general.py --data /mnt/personal/qiankangan/$sensor_type/val \
# --model /mnt/personal/qiankangan/logs/MotionNet/at128_0/train_multi_seq/2024-09-14_17-36-27/epoch_50.pth \
# --sensor_type $sensor_type \
# --split test --log logs_eval/test_at128_0 --bs 1 --net MotionNet \
# --use_v --debug

# nuscenes
# /mnt/personal/qiankangan/logs/GRUVAE_bs2_MotionNet/train_multi_seq/2024-10-31_11-14-10/save_model.pth
# python eval_general.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/GRUVAE_MotionNet/train_multi_seq/2024-10-25_23-06-12/save_model.pth \
# --sensor_type $sensor_type --net GRUMotionNetVAE \
# --split test --log ./logs/GRUgen_bs4_$sensor_type --bs 1

# python eval.py --data /mnt/personal/qiankangan/test \
# --model /mnt/personal/qiankangan/logs/GRUVAE_MotionNet/train_multi_seq/2024-10-25_23-06-12/save_model.pth \
# --sensor_type nuscenes --net GRUMotionNetVAE \
# --split test --log ./logs/GRUgen_bs4_$sensor_type --bs 1

# /mnt/personal/qiankangan/logs/GRUVAE_origin_bs4_MotionNet/train_multi_seq/2024-11-08_17-18-28/save_model.pth
python eval.py --data /mnt/personal/qiankangan/test \
--model /mnt/personal/qiankangan/logs/GRUVAE_MotionNet/train_multi_seq/2024-10-25_23-06-12/save_model.pth \
--sensor_type $sensor_type --net GRUMotionNetVAE \
--split test --log ./logs/GRUgen_origin_bs4_$sensor_type --bs 1 --general