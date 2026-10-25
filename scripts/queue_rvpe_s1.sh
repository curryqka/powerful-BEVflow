#!/bin/bash
# 排队启动 RVpE s1：RVpE 单 run 需 ~28GB，无法与现有 run 共卡。
# 等 GPU0 上的 fuser 与 rvpe_s0 全部结束后启动，并把 run 加入看守规格。
set -u
CODE=/home/kaneqian/BEVflow
EXP=/media/nas_users/kaneqian/Experiment/bevflow
VENV=$EXP/_env/bevflow/bin/python
LOG=$EXP/_cache/cmdlogs/queue_rvpe_s1.log
export BEVFLOW_DATA=/media/nas_users/kaneqian/data/bevflow

echo "$(date) queue started, waiting for GPU0 empty" >> $LOG
while pgrep -f "run_id=20261023_PM_fuser_s0|run_id=20261023_PM_rvpe_s0" > /dev/null; do
    sleep 300
done
echo "$(date) GPU0 empty, launching rvpe_s1" >> $LOG
cd $CODE
setsid nohup env CUDA_VISIBLE_DEVICES=0 $VENV scripts/train.py \
    --config configs/exp/priormotion_rvpe.yaml \
    --set run.run_id=20261023_PM_rvpe_s1 train.seed=1 train.val_interval=2 \
    >> $EXP/_cache/cmdlogs/gpu0_pm_rvpe_s1.log 2>&1 &
sleep 60
echo "20261023_PM_rvpe_s1|0|priormotion_rvpe.yaml|train.seed=1 train.val_interval=2|45" >> $EXP/_cache/watchdog_runs.txt
echo "$(date) rvpe_s1 launched and added to watchdog" >> $LOG
