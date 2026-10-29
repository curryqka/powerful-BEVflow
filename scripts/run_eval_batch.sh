#!/bin/bash
# 一次性评测批：ext55 三个 test 点 + D8 扩展口径四缺项。7 个并发，wait 汇总。
set -u
EXP=/media/nas_users/kaneqian/Experiment/bevflow
VENV=$EXP/_env/bevflow/bin/python
RES=/media/nas_users/kaneqian/results/bevflow
DATA=/media/nas_users/kaneqian/data/bevflow/preprocessed/motionnet_v1
LOG=$EXP/_cache/cmdlogs
export BEVFLOW_DATA=/media/nas_users/kaneqian/data/bevflow
cd /home/kaneqian/BEVflow

run() {
    local name=$1; shift
    CUDA_VISIBLE_DEVICES=1 "$@" > "$LOG/batch_$name.log" 2>&1 &
    echo "started $name pid $!"
}

run ext55_e55 $VENV scripts/eval.py --model priormotion \
    --ckpt $EXP/P6RV/20261028_PM_rvpe_s1_ext55/ckpt/epoch_55.pth \
    --data-root $DATA/test --split test --out-dir $RES/P6/eval/PM_rvpe_s1_ext55_test \
    --run-id PM_rvpe_s1_ext55_test --bs 1 --worker 2
for ep in 46 50; do
    run ext55_e$ep $VENV scripts/eval.py --model priormotion \
        --ckpt $EXP/P6RV/20261028_PM_rvpe_s1_ext55/ckpt/epoch_$ep.pth \
        --data-root $DATA/test --split test --out-dir $RES/P6/eval/ext55_ep$ep \
        --run-id ext55_ep$ep --bs 4 --worker 2
done
run d8_DSpG_s0 $VENV scripts/eval_general.py --model priormotion \
    --ckpt $EXP/P6PM/20261018_PM_DSpG_s0/ckpt/epoch_45.pth \
    --data-root $DATA/test --split test --out-dir $RES/P6/eval/DSpG_s0_general --run-id DSpG_s0_general
run d8_DSpG_s1 $VENV scripts/eval_general.py --model priormotion \
    --ckpt $EXP/P6PM/20261018_PM_DSpG_s1/ckpt/epoch_45.pth \
    --data-root $DATA/test --split test --out-dir $RES/P6/eval/DSpG_s1_general --run-id DSpG_s1_general
run d8_fuser $VENV scripts/eval_general.py --model priormotion \
    --ckpt $EXP/P6EXP/20261023_PM_fuser_s0/ckpt/epoch_45.pth \
    --data-root $DATA/test --split test --out-dir $RES/P6/eval/fuser_general --run-id fuser_general
run d8_rvpe_s1 $VENV scripts/eval_general.py --model priormotion \
    --ckpt $EXP/P6RV/20261023_PM_rvpe_s1/ckpt/epoch_45.pth \
    --data-root $DATA/test --split test --out-dir $RES/P6/eval/PM_rvpe_s1_test_general --run-id PM_rvpe_s1_general

wait
echo "ALL_EVALS_DONE $(date)"
