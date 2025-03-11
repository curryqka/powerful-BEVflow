RUN_ROOT="/SHARE/qiankangan/bevflow"
# python train_VAE.py --data $RUN_ROOT/input-data --evaldata $RUN_ROOT/val \
# --batch 2 --nepoch 45 --nworker 4 --lr 0.0016 \
# --use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
# --log --auxiliary --use_adj_frame_pred --model UniGridInsGRUMotionNetVAE \
# --logpath $RUN_ROOT/logs/PriorMotionV0_test --general --gpu 3 \
# --sensor_type nuscenes --grid_prior --debug
# # --resume /SHARE/qiankangan/bevflow/logs/UniInsGridGRUVAE_bs4_MotionNet/train_multi_seq/2025-01-17_04-32-23/


model=$1
python train_PriorMotion.py --data $RUN_ROOT/input-data --evaldata $RUN_ROOT/test \
--batch 2 --nepoch 45 --nworker 4 --lr 0.0016 \
--use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
--log --auxiliary --use_adj_frame_pred --model $model \
--logpath $RUN_ROOT/logs/${model}_test --gpu 0 \
--sensor_type nuscenes --ins_prior --grid_prior
# --pre_model_path /SHARE/qiankangan/bevflow/logs/MotionNet/train_multi_seq/2025-02-14_08-48-16/epoch_24.pth
# --resume /SHARE/qiankangan/bevflow/logs/UniInsGridGRUVAE_bs4_MotionNet/train_multi_seq/2025-01-17_04-32-23/
