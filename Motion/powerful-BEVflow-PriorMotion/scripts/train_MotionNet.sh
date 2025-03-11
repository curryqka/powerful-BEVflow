RUN_ROOT="/SHARE/qiankangan/bevflow"

python train_multi_seq.py --data $RUN_ROOT/input-data --evaldata $RUN_ROOT/val \
--batch 8 --nepoch 80 --nworker 4 --lr 0.0016 \
--use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
--log --model MotionNet \
--logpath $RUN_ROOT/logs/MotionNet_debug --gpu 0 # --resume /SHARE/qiankangan/bevflow/logs/InsMotionNet_withoutselfAttn/train_multi_seq/2025-02-08_12-49-56
# --pre_model_path /SHARE/qiankangan/bevflow/logs/InsMotionNet_withoutselfAttn/train_multi_seq/2025-02-08_12-49-56/latest.pth
#
