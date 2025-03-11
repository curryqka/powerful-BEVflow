RUN_ROOT="/SHARE/qiankangan/bevflow"

python train_multi_seq.py --data $RUN_ROOT/input-data --evaldata $RUN_ROOT/val \
--batch 2 --nepoch 45 --nworker 4 --lr 0.0016 \
--use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
--log --model InsMotionNet \
--logpath $RUN_ROOT/logs/test_mask_fn05_lr0016 --gpu 1 --mask
-resume /SHARE/qiankangan/bevflow/logs/test_mask/train_multi_seq/2025-02-12_16-05-15
