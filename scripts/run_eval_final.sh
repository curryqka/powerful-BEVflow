#!/bin/bash
# 剩余评测（ext55 e50 + D8 五个），数据用 /tmp 本地副本，3 并发 × 2 波。
set -u
EXP=/media/nas_users/kaneqian/Experiment/bevflow
VENV=$EXP/_env/bevflow/bin/python
RES=/media/nas_users/kaneqian/results/bevflow
DATA=/tmp/bevflow_data/test
LOG=$EXP/_cache/cmdlogs
export BEVFLOW_DATA=/media/nas_users/kaneqian/data/bevflow
cd /home/kaneqian/BEVflow

w() { "$@" & }

w env CUDA_VISIBLE_DEVICES=1 $VENV scripts/eval.py --model priormotion \
  --ckpt $EXP/P6RV/20261028_PM_rvpe_s1_ext55/ckpt/epoch_50.pth \
  --data-root $DATA --split test --out-dir $RES/P6/eval/ext55_ep50 \
  --run-id ext55_ep50 --bs 2 --worker 2 > $LOG/final_ext55_e50.log 2>&1
w env CUDA_VISIBLE_DEVICES=1 $VENV scripts/eval_general.py --model priormotion \
  --ckpt $EXP/P6PM/20261018_PM_DSpG_s0/ckpt/epoch_45.pth \
  --data-root $DATA --split test --out-dir $RES/P6/eval/DSpG_s0_general --run-id DSpG_s0_general \
  > $LOG/final_d8_DSpG_s0.log 2>&1
w env CUDA_VISIBLE_DEVICES=1 $VENV scripts/eval_general.py --model priormotion \
  --ckpt $EXP/P6PM/20261018_PM_DSpG_s1/ckpt/epoch_45.pth \
  --data-root $DATA --split test --out-dir $RES/P6/eval/DSpG_s1_general --run-id DSpG_s1_general \
  > $LOG/final_d8_DSpG_s1.log 2>&1
wait

w env CUDA_VISIBLE_DEVICES=1 $VENV scripts/eval_general.py --model priormotion \
  --ckpt $EXP/P6EXP/20261023_PM_fuser_s0/ckpt/epoch_45.pth \
  --data-root $DATA --split test --out-dir $RES/P6/eval/fuser_general --run-id fuser_general \
  > $LOG/final_d8_fuser.log 2>&1
w env CUDA_VISIBLE_DEVICES=1 $VENV scripts/eval_general.py --model priormotion \
  --ckpt $EXP/P6RV/20261023_PM_rvpe_s1/ckpt/epoch_45.pth \
  --data-root $DATA --split test --out-dir $RES/P6/eval/PM_rvpe_s1_test_general --run-id PM_rvpe_s1_general \
  > $LOG/final_d8_rvpe_s1.log 2>&1
w env CUDA_VISIBLE_DEVICES=1 $VENV scripts/eval_general.py --model priormotion \
  --ckpt $EXP/P6RV/20261023_PM_rvpe_s0/ckpt/epoch_45.pth \
  --data-root $DATA --split test --out-dir $RES/P6/eval/PM_rvpe_s0_test_general --run-id PM_rvpe_s0_general \
  > $LOG/final_d8_rvpe_s0.log 2>&1
wait
echo "FINAL_EVALS_DONE $(date)"
