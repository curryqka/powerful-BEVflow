#!/bin/bash
# 剩余 6 个评测，3 并发 × 2 波（NAS 负载受限，不再 7 路齐发）。
set -u
EXP=/media/nas_users/kaneqian/Experiment/bevflow
VENV=$EXP/_env/bevflow/bin/python
RES=/media/nas_users/kaneqian/results/bevflow
DATA=/media/nas_users/kaneqian/data/bevflow/preprocessed/motionnet_v1
LOG=$EXP/_cache/cmdlogs
export BEVFLOW_DATA=/media/nas_users/kaneqian/data/bevflow
cd /home/kaneqian/BEVflow

wave() {
    "$@" &
}

wave env CUDA_VISIBLE_DEVICES=1 $VENV scripts/eval.py --model priormotion \
    --ckpt $EXP/P6RV/20261028_PM_rvpe_s1_ext55/ckpt/epoch_46.pth \
    --data-root $DATA/test --split test --out-dir $RES/P6/eval/ext55_ep46 \
    --run-id ext55_ep46 --bs 4 --worker 2 > $LOG/batch_ext55_e46.log 2>&1
wave env CUDA_VISIBLE_DEVICES=1 $VENV scripts/eval.py --model priormotion \
    --ckpt $EXP/P6RV/20261028_PM_rvpe_s1_ext55/ckpt/epoch_50.pth \
    --data-root $DATA/test --split test --out-dir $RES/P6/eval/ext55_ep50 \
    --run-id ext55_ep50 --bs 4 --worker 2 > $LOG/batch_ext55_e50.log 2>&1
wave env CUDA_VISIBLE_DEVICES=1 $VENV scripts/eval_general.py --model priormotion \
    --ckpt $EXP/P6PM/20261018_PM_DSpG_s0/ckpt/epoch_45.pth \
    --data-root $DATA/test --split test --out-dir $RES/P6/eval/DSpG_s0_general --run-id DSpG_s0_general \
    > $LOG/batch_d8_DSpG_s0.log 2>&1
wait

wave env CUDA_VISIBLE_DEVICES=1 $VENV scripts/eval_general.py --model priormotion \
    --ckpt $EXP/P6PM/20261018_PM_DSpG_s1/ckpt/epoch_45.pth \
    --data-root $DATA/test --split test --out-dir $RES/P6/eval/DSpG_s1_general --run-id DSpG_s1_general \
    > $LOG/batch_d8_DSpG_s1.log 2>&1
wave env CUDA_VISIBLE_DEVICES=1 $VENV scripts/eval_general.py --model priormotion \
    --ckpt $EXP/P6EXP/20261023_PM_fuser_s0/ckpt/epoch_45.pth \
    --data-root $DATA/test --split test --out-dir $RES/P6/eval/fuser_general --run-id fuser_general \
    > $LOG/batch_d8_fuser.log 2>&1
wave env CUDA_VISIBLE_DEVICES=1 $VENV scripts/eval_general.py --model priormotion \
    --ckpt $EXP/P6RV/20261023_PM_rvpe_s1/ckpt/epoch_45.pth \
    --data-root $DATA/test --split test --out-dir $RES/P6/eval/PM_rvpe_s1_test_general --run-id PM_rvpe_s1_general \
    > $LOG/batch_d8_rvpe_s1.log 2>&1
wait
echo "REST_EVALS_DONE $(date)"
