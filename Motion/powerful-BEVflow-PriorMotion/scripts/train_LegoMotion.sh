RUN_ROOT="/SHARE/qiankangan/bevflow"
mask_weight=1.0
cell_class_weight=1.0
disp_weight=1.0
motion_state_weight=1.0
lr=$1
python train_MaskBEVflow.py --data $RUN_ROOT/input-data --evaldata $RUN_ROOT/val \
--batch 8 --nepoch 45 --nworker 4 --lr $lr \
--use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
--mask_weight $mask_weight --cell_class_weight $cell_class_weight --disp_weight $disp_weight --motion_state_weight $motion_state_weight \
--log --model LegoMotion \
--logpath $RUN_ROOT/logs/test_LegoMotion_${mask_weight}_${cell_class_weight}_${disp_weight}_${motion_state_weight}_${lr} --gpu 0 --mask \
# --resume /SHARE/qiankangan/bevflow/logs/InsMotionNetv2_1.5_1.0_1.0_1.0/train_multi_seq/2025-02-16_15-01-05/latest.pth
