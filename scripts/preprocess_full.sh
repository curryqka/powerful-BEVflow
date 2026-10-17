#!/bin/bash
# 全量预处理：train/val/test 分片并行，完成后拍平 val/test 供评测。
# 用法：bash scripts/preprocess_full.sh [N_SHARDS_TRAIN]（默认 4）
set -u
CODE=/home/kaneqian/BEVflow
DATA=/media/nas_users/kaneqian/data/bevflow
EXP=/media/nas_users/kaneqian/Experiment/bevflow
VENV=$EXP/_env/bevflow/bin/python
OUT=$DATA/preprocessed/motionnet_v1
LOGDIR=$EXP/_data_logs/preprocess_full
NT=${1:-4}
mkdir -p $LOGDIR $OUT/{train,val,test}

scene_ids() {  # scene_ids <split>
    $VENV -c "
import numpy as np
s = np.load('$DATA/splits/split.npy', allow_pickle=True).item()
print(' '.join(str(int(x.split('_')[1])) for x in s['$1']))"
}

shard_run() {  # shard_run <split> <shard_idx> <n_shards> <ids...>
    local split=$1 idx=$2 n=$3; shift 3
    local ids=("$@")
    local total=${#ids[@]}
    local per=$(( (total + n - 1) / n ))
    local start=$(( idx * per ))
    local len=$per
    if [ $((start + len)) -gt $total ]; then len=$(( total - start )); fi
    if [ $start -ge $total ] || [ $len -le 0 ]; then return 0; fi
    local sub=("${ids[@]:$start:$len}")
    echo "[shard $split/$idx] scenes: ${sub[0]}..${sub[$(( ${#sub[@]} - 1 ))]} (${#sub[@]})"
    $VENV $CODE/scripts/gen_data.py --split $split --scenes ${sub[@]} \
        --savepath $OUT/$split > $LOGDIR/${split}_$idx.log 2>&1
    echo "[shard $split/$idx] exit $?"
}

TRAIN_IDS=($(scene_ids train))
VAL_IDS=($(scene_ids val))
TEST_IDS=($(scene_ids test))
echo "train=${#TRAIN_IDS[@]} val=${#VAL_IDS[@]} test=${#TEST_IDS[@]}"

PIDS=()
for i in $(seq 0 $((NT - 1))); do
    shard_run train $i $NT "${TRAIN_IDS[@]}" &
    PIDS+=($!)
done
NV=2
for i in $(seq 0 $((NV - 1))); do
    shard_run val $i $NV "${VAL_IDS[@]}" &
    PIDS+=($!)
done
NTS=2
for i in $(seq 0 $((NTS - 1))); do
    shard_run test $i $NTS "${TEST_IDS[@]}" &
    PIDS+=($!)
done

FAIL=0
for p in "${PIDS[@]}"; do wait $p || FAIL=1; done
echo "all shards done, FAIL=$FAIL"

# 拍平 val/test（DatasetSingleSeq 只认扁平文件）
$VENV $CODE/scripts/flatten_eval_data.py $OUT/val
$VENV $CODE/scripts/flatten_eval_data.py $OUT/test
echo "PREPROCESS_FULL_DONE FAIL=$FAIL"
du -sh $OUT/*
exit $FAIL
