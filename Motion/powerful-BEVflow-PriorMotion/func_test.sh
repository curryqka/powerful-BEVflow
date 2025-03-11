export PYTHONPATH=/home/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH

sensor_type=$1
GPU=$2
# FMCW training mixmodels

# train FMCW base model
# python train_multi_seq_wt128_mix.py --data /mnt/personal/qiankangan/all_$sensor_type/train \
# --evaldata /mnt/personal/qiankangan/all_$sensor_type/val \
# --batch 8 --lr 0.0016 --nepoch 2 --nworker 2  --log --logpath /home/qiankangan/logs/test_$sensor_type --gpu $GPU \
# --model MixGeneralMotionNet \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 \
# --use_v --debug

# train FMCW noabsv model
# python train_multi_seq_wt128_mix.py --data /mnt/personal/qiankangan/all_$sensor_type/train \
# --evaldata /mnt/personal/qiankangan/all_$sensor_type/val \
# --batch 8 --lr 0.0016 --nepoch 2 --nworker 2  --log --logpath /home/qiankangan/logs/test_withnoabsv_$sensor_type --gpu $GPU \
# --model MixGeneralMotionNet \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 \
# --debug

# train FMCW withabsv withdir model
# python train_multi_seq_wt128_mix.py --data /mnt/personal/qiankangan/all_$sensor_type/train \
# --evaldata /mnt/personal/qiankangan/all_$sensor_type/val \
# --batch 8 --lr 0.0016 --nepoch 2 --nworker 2  --log --logpath /home/qiankangan/logs/test_withdir_$sensor_type --gpu $GPU \
# --model MixGeneralMotionNet \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 \
# --use_v --use_vdir --debug

# -------------------------------------------------------------------------------------------
# FMCW new channels model
# train FMCW base model
# python train_multi_seq_wt128.py --data /mnt/personal/qiankangan/all_$sensor_type/train \
# --evaldata /mnt/personal/qiankangan/all_$sensor_type/val \
# --batch 8 --lr 0.0016 --nepoch 2 --nworker 2  --log --logpath /home/qiankangan/logs/testbase_$sensor_type --gpu $GPU \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 \
# --use_v --debug

# FMCW withnoabsv training
# python train_multi_seq_wt128.py --data /mnt/personal/qiankangan/all_$sensor_type/train \
# --evaldata /mnt/personal/qiankangan/all_$sensor_type/val \
# --batch 8 --lr 0.0016 --nepoch 2 --nworker 2  --log --logpath /home/qiankangan/logs/testbase_$sensor_type --gpu $GPU \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 \
# --debug

# FMCW withdir training
# python train_multi_seq_wt128.py --data /mnt/personal/qiankangan/all_$sensor_type/train \
# --evaldata /mnt/personal/qiankangan/all_$sensor_type/val \
# --batch 8 --lr 0.0016 --nepoch 2 --nworker 2  --log --logpath /home/qiankangan/logs/testbase_$sensor_type --gpu $GPU \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 \
# --use_v --use_vdir --debug

# -------------------------------------------------------------------------------------------
# FMCW evaluation general 
# FMCW mixmodels
# python eval_general.py --data /mnt/personal/qiankangan/$sensor_type/val \
# --model /home/qiankangan/logs/test_wt128_0/train_multi_seq/2024-10-18_15-43-53/save_model.pth \
# --split test --log ../logs_eval/test_wt128_0 --bs 1 --net MixGeneralMotionNet \
# --sensor_type $sensor_type \
# --use_v --debug

# FMCW mixmodels for mask 
# python eval_general.py --data /mnt/personal/qiankangan/$sensor_type/val \
# --model /home/qiankangan/logs/test_wt128_0/train_multi_seq/2024-10-18_15-43-53/save_model.pth \
# --split test --log ../logs_eval/test_wt128_0_mask --bs 1 --net MixGeneralMotionNet \
# --sensor_type $sensor_type \
# --use_v --general --debug

# FMCW mixmodels withnoabsv
# python eval_general.py --data /mnt/personal/qiankangan/$sensor_type/val \
# --model /home/qiankangan/logs/test_withdir_wt128_0/train_multi_seq/2024-10-18_15-56-11/save_model.pth \
# --split test --log ../logs_eval/testwithdir_wt128_0 --bs 1 --net MixGeneralMotionNet \
# --sensor_type $sensor_type \
# --use_v --use_vdir --debug

# -------------------------------------------------------------------------------------------
# FMCW new channels model
# python eval_general.py --data /mnt/personal/qiankangan/$sensor_type/val \
# --model /home/qiankangan/logs/testbase_wt128_0/train_multi_seq/2024-10-18_15-57-23/save_model.pth \
# --split test --log ../logs_eval/testbase_wt128_0 --bs 1 --net MotionNet \
# --sensor_type $sensor_type \
# --use_v --debug

# FMCW new channels model withnoabsv
# python eval_general.py --data /mnt/personal/qiankangan/$sensor_type/val \
# --model /home/qiankangan/logs/testbase_wt128_0/train_multi_seq/2024-10-18_15-59-56/save_model.pth \
# --split test --log ../logs_eval/testbasenoabsv_wt128_0 --bs 1 --net MotionNet \
# --sensor_type $sensor_type \
# --debug

# FMCW new channels model withdir
# python eval_general.py --data /mnt/personal/qiankangan/$sensor_type/val \
# --model /home/qiankangan/logs/testbase_wt128_0/train_multi_seq/2024-10-18_16-00-58/save_model.pth \
# --split test --log ../logs_eval/testbasewithdir_wt128_0 --bs 1 --net MotionNet \
# --sensor_type $sensor_type \
# --use_v --use_vdir --debug

# -------------------------------------------------------------------------------------------
# FMCW evaluation base
# FMCW mixmodel
# python eval.py --data /mnt/personal/qiankangan/$sensor_type/val \
# --model /home/qiankangan/logs/test_wt128_0/train_multi_seq/2024-10-18_15-43-53/save_model.pth \
# --split test --log ../logs_eval/basetest_wt128_0 --bs 1 --net MixGeneralMotionNet \
# --sensor_type $sensor_type \
# --use_v --debug

# FMCW mixmodel withnoabsv
# python eval.py --data /mnt/personal/qiankangan/$sensor_type/val \
# --model /home/qiankangan/logs/test_withnoabsv_wt128_0/train_multi_seq/2024-10-18_15-50-32/save_model.pth \
# --split test --log ../logs_eval/basetestnoabsv_wt128_0 --bs 1 --net MixGeneralMotionNet \
# --sensor_type $sensor_type \
# --debug

# FMCW mixmodel withabsvdir
# python eval.py --data /mnt/personal/qiankangan/$sensor_type/val \
# --model /home/qiankangan/logs/test_withdir_wt128_0/train_multi_seq/2024-10-18_15-56-11/save_model.pth \
# --split test --log ../logs_eval/basetestwithabsv_wt128_0 --bs 1 --net MixGeneralMotionNet \
# --sensor_type $sensor_type \
# --use_v --use_vdir --debug --general

# -------------------------------------------------------------------------------------------
# FMCW new channels base model
# python eval.py --data /mnt/personal/qiankangan/$sensor_type/val \
# --model /home/qiankangan/logs/testbase_wt128_0/train_multi_seq/2024-10-18_15-57-23/save_model.pth \
# --split test --log ../logs_eval/basetestbase_wt128_0 --bs 1 --net MotionNet \
# --sensor_type $sensor_type \
# --use_v --debug 

# FMCW new channels withnoabsv
# python eval.py --data /mnt/personal/qiankangan/$sensor_type/val \
# --model /home/qiankangan/logs/testbase_wt128_0/train_multi_seq/2024-10-18_15-59-56/save_model.pth \
# --split test --log ../logs_eval/basetestnoabsv_wt128_0 --bs 1 --net MotionNet \
# --sensor_type $sensor_type \
# --debug --general

# FMCW new channels withdir
# python eval.py --data /mnt/personal/qiankangan/$sensor_type/val \
# --model /home/qiankangan/logs/testbase_wt128_0/train_multi_seq/2024-10-18_16-00-58/save_model.pth \
# --split test --log ../logs_eval/basetestwithdir_wt128_0 --bs 1 --net MotionNet \
# --sensor_type $sensor_type \
# --use_v --use_vdir --debug

# -------------------------------------------------------------------------------------------
# GenMotionNet
python train_VAE.py --data /mnt/personal/qiankangan/input-data --evaldata /mnt/personal/qiankangan/val \
--batch 8 --nepoch 60 --nworker 2 --pre_model_path /mnt/personal/qiankangan/weak_WeakMotionNet.pth \
--use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
--log --auxiliary --use_adj_frame_pred --model VAE_MotionNet \
--logpath ../logs/VAE_MotionNet_test_$sensor_type --general --gpu $GPU \
--debug



