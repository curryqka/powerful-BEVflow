RUN_ROOT="/SHARE/qiankangan/bevflow"
cd /workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion
pip install -r requirements.txt
model=$1
lr=0.0016

python train_VAE.py --data $RUN_ROOT/input-data --evaldata $RUN_ROOT/test \
--batch 2 --nepoch 45 --nworker 4 --lr $lr \
--use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
--log --auxiliary --use_adj_frame_pred --model $model \
--logpath $RUN_ROOT/logs/${model}_${lr}_ab_GRU --general --gpu 0 \
--sensor_type nuscenes --grid_prior --ins_prior
