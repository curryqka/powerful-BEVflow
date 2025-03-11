model=$1
RUN_ROOT="/SHARE/qiankangan/bevflow"
# python eval_general.py --data $RUN_ROOT/test \
# --model $RUN_ROOT/logs/MotionNet_GRU/train_multi_seq/2025-01-24_10-22-25/save_model.pth \
# --sensor_type nuscenes --net $model \
# --split test --log $RUN_ROOT/logs/eval_${model}_nuscenes_general --bs 1
# # --general

# python eval_general.py --data $RUN_ROOT/test \
# --model $RUN_ROOT/logs/MotionNet_CNN/train_multi_seq/2025-01-24_09-59-24/save_model.pth \
# --sensor_type nuscenes --net $model \
# --split test --log $RUN_ROOT/logs/eval_${model}_nuscenes_general --bs 1
# # # --general

python eval_general.py --data $RUN_ROOT/test \
--model $RUN_ROOT/logs/MotionNetHigher/train_multi_seq/2025-01-25_03-48-22/save_model.pth \
--sensor_type nuscenes --net $model \
--split test --log $RUN_ROOT/logs/eval_${model}_nuscenes_general --bs 1
# # --general

# python eval.py --data $RUN_ROOT/test \
# --model $RUN_ROOT/logs/UniGridGenMotionNet_grid/train_multi_seq/2025-01-25_08-06-10/save_model.pth \
# --sensor_type nuscenes --net $model \
# --split test --log $RUN_ROOT/logs/eval_${model}_nuscenes --bs 1
# # --general
