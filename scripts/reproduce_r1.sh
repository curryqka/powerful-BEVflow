#!/bin/bash
# R1 全流程一条命令复现：数据生产 → 训练 → last-10 评测 → test 终评 → 可视化。
# 用法：RUN_ID=<date>_R1B_motionnet_s0 bash scripts/reproduce_r1.sh [N_SHARDS]
# 耗时：数据 ~1.2h + 训练 ~14h + 评测 ~1.5h。断点安全：训练支持 --resume auto。
set -eu

CODE=/home/kaneqian/BEVflow
DATA=/media/nas_users/kaneqian/data/bevflow
RESULTS=/media/nas_users/kaneqian/results/bevflow
EXP=/media/nas_users/kaneqian/Experiment/bevflow
VENV=$EXP/_env/bevflow/bin/python
RUN_ID=${RUN_ID:?need RUN_ID, e.g. 20261018_R1B_motionnet_s0}
NT=${1:-4}

echo "== stage 1: 数据生产（已存在则跳过）=="
if [ -f $DATA/preprocessed/motionnet_v1/data_card.json ]; then
    echo "data exists, skip"
else
    bash $CODE/scripts/preprocess_full.sh $NT
fi

echo "== stage 2: 训练 R1B =="
CUDA_VISIBLE_DEVICES=0 $VENV $CODE/scripts/train.py \
    --config $CODE/configs/exp/R1B_budget.yaml \
    --set run.run_id=$RUN_ID

echo "== stage 3: last-10 评测 =="
CUDA_VISIBLE_DEVICES=0 $VENV $CODE/scripts/eval_per_ckpt.py \
    --run-dir $EXP/R1B/$RUN_ID \
    --val-root $DATA/preprocessed/motionnet_v1/val

echo "== stage 4: test 终评（bs1）=="
CUDA_VISIBLE_DEVICES=0 $VENV $CODE/scripts/eval.py \
    --ckpt $EXP/R1B/$RUN_ID/ckpt/epoch_45.pth \
    --data-root $DATA/preprocessed/motionnet_v1/test \
    --split test --out-dir $RESULTS/R1/eval/${RUN_ID}_test \
    --run-id ${RUN_ID}_test --bs 1 --worker 4

echo "== stage 5: 可视化 =="
CUDA_VISIBLE_DEVICES=0 $VENV $CODE/scripts/viz.py \
    --val-root $DATA/preprocessed/motionnet_v1/val \
    --ckpt $EXP/R1B/$RUN_ID/ckpt/epoch_45.pth \
    --out-dir $RESULTS/R1/viz/$RUN_ID

echo "REPRODUCE_R1_DONE: $RUN_ID"
