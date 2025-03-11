RUN_ROOT="/SHARE/qiankangan/bevflow"
cd /workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion
# pip install -r requirements.txt
# python eval_general.py --data $RUN_ROOT/test \
# --model $RUN_ROOT/logs/MotionNet_GRU/train_multi_seq/2025-01-24_10-22-25/latest.pth \
# --sensor_type nuscenes --net $model \
# --split test --log $RUN_ROOT/logs/eval_${model}_latest_general --bs 1
# --general

# python eval_general.py --data $RUN_ROOT/test \
# --model $RUN_ROOT/logs/MotionNet_CNN/train_multi_seq/2025-01-24_09-59-24/latest.pth \
# --sensor_type nuscenes --net $model \
# --split test --log $RUN_ROOT/logs/eval_${model}_latest_general --bs 1
# # --general

# python eval_general.py --data $RUN_ROOT/test \
# --model $RUN_ROOT/logs/MotionNetHigher/train_multi_seq/2025-01-25_03-48-22/latest.pth \
# --sensor_type nuscenes --net $model \
# --split test --log $RUN_ROOT/logs/eval_${model}_latest_general --bs 1
# # --general

# python eval.py --data $RUN_ROOT/test \
# --model $RUN_ROOT/logs/UniGridGenMotionNet_grid/train_multi_seq/2025-01-25_08-06-10/latest.pth \
# --sensor_type nuscenes --net $model \
# --split test --log $RUN_ROOT/logs/eval_${model}_latest --bs 1
# # --general


# multi_GPU MotionNet
# python eval.py --data $RUN_ROOT/test \
# --model $RUN_ROOT/logs/test_multigpu_MotionNet/train_multi_seq/2025-02-14_08-57-26/save_model.pth \
# --sensor_type nuscenes --net $model \
# --split test --log $RUN_ROOT/logs/eval_multi_${model} --bs 1

# InsMotionNet
# python eval_general.py --data $RUN_ROOT/val \
# --model $RUN_ROOT/logs/test_multigpu_mask_fn05_lr0016/train_multi_seq/2025-02-14_03-11-38/epoch_4.pth \
# --sensor_type nuscenes --net $model \
# --split test --log $RUN_ROOT/logs/eval_${model} --bs 1

# # /SHARE/qiankangan/bevflow/logs/gridinsMotionNet/train_multi_seq/2025-02-21_06-32-30/save_model.pth
# python eval.py --data $RUN_ROOT/test \
# --model $RUN_ROOT/logs/gridinsMotionNet/train_multi_seq/2025-02-21_06-32-30/epoch_34.pth \
# --sensor_type nuscenes --net $model \
# --split test --log $RUN_ROOT/logs/eval_${model} --bs 1

# /SHARE/qiankangan/bevflow/logs/gridinsMotionNetSE_0.0016_8/train_multi_seq/2025-02-25_12-53-09/epoch_32.pth
# python eval.py --data $RUN_ROOT/test \
# --model $RUN_ROOT/logs/gridinsMotionNetSE_0.0016_8/train_multi_seq/2025-02-25_12-53-09/save_model.pth \
# --sensor_type nuscenes --net $model \
# --split test --log $RUN_ROOT/logs/eval_${model} --bs 1

model="UniGridInsGRUMotionNetVAE"
# PriorMotionV0 /SHARE/qiankangan/bevflow/logs/UniInsGridGRUVAE_bs4_MotionNet/train_multi_seq/2025-01-17_04-32-23/save_model.pth
python eval_general.py --data $RUN_ROOT/test \
--model $RUN_ROOT/logs/UniInsGridGRUVAE_bs4_MotionNet/train_multi_seq/2025-01-17_04-32-23/save_model.pth \
--sensor_type nuscenes --net $model \
--split test --log $RUN_ROOT/logs/eval_${model}_general --bs 1
