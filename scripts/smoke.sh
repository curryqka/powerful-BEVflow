#!/bin/bash
# 五步冒烟：S1 数据，S2 训练(短训+过拟合)，S3 存取，S4 评测，S5 可视化。
# 用法：SMOKE_MODE=upstream|new bash scripts/smoke.sh
#   upstream：S2/S4 调上游脚本（重构前基线）
#   new：     S2/S4 调 bevflow 新入口（重构后），S1 数据检查共用，检查项完全相同
set -u

CODE=/home/kaneqian/BEVflow
DATA=/media/nas_users/kaneqian/data/bevflow
RESULTS=/media/nas_users/kaneqian/results/bevflow
EXP=/media/nas_users/kaneqian/Experiment/bevflow
VENV=$EXP/_env/bevflow/bin/python
UPSTREAM=$CODE/third_party/MotionNet
MODE=${SMOKE_MODE:-upstream}

TRAIN_ROOT=$DATA/preprocessed/mini/train
VAL_ROOT=$DATA/preprocessed/mini/val
export PYTHONPATH=$UPSTREAM:$UPSTREAM/nuscenes-devkit/python-sdk
export CUDA_VISIBLE_DEVICES=0

TS=$(date +%Y%m%d_%H%M%S)
SMOKE=$RESULTS/_smoke/${MODE}_$TS
mkdir -p $SMOKE/{train,eval,viz}
git -C $CODE rev-parse HEAD > $SMOKE/git.txt
git -C $CODE status --short > $SMOKE/git_dirty.txt

declare -A STATUS

run_step() {  # run_step <name> <logfile> <cmd...>
    local name=$1 log=$2; shift 2
    echo "[$(date +%H:%M:%S)] $name start"
    if "$@" > "$log" 2>&1; then
        STATUS[$name]=PASS; tail -2 "$log" | sed 's/^/    /'
    else
        STATUS[$name]=FAIL; echo "    see $log"
    fi
}

# S1 数据（两种模式共用：同一批数据文件，同一检查）
if run_step S1 $SMOKE/s1.log $VENV $CODE/scripts/smoke_checks.py s1 \
        --train-root $TRAIN_ROOT --val-root $VAL_ROOT; then S1_OK=1; else S1_OK=0; fi

# S2a 短训（2 epoch ≈ 104 iter）
S2A_OK=0
RUN_DIR=""
if [ $S1_OK -eq 1 ]; then
    if [ "$MODE" = new ]; then
        (
            cd $CODE && $VENV scripts/train.py --config configs/exp/smoke_mini.yaml \
                --set run.run_id=smoke_$TS
        ) > $SMOKE/s2a_train.log 2>&1
        RUN_DIR=$EXP/SMOKE/smoke_$TS
        if [ -f "$RUN_DIR/ckpt/epoch_2.pth" ] && grep -q '"type": "epoch", "epoch": 2' "$RUN_DIR/logs/train.jsonl" 2>/dev/null; then
            STATUS[S2a_short_train]=PASS; S2A_OK=1
        else
            STATUS[S2a_short_train]=FAIL; echo "    see $SMOKE/s2a_train.log"
        fi
    else
        mkdir -p $SMOKE/train/run1
        # 注意：上游 --logpath 会在其下再建 train_multi_seq/<时间戳>/，ckpt 与 log.txt 都在里面
        (
            cd $UPSTREAM &&
            $VENV train_multi_seq.py -d $TRAIN_ROOT --batch 4 --nepoch 2 --nworker 2 \
                --log --logpath $SMOKE/train/run1
        ) > $SMOKE/s2a_train.log 2>&1
        RUN_DIR=$(ls -td $SMOKE/train/run1/train_multi_seq/*/ 2>/dev/null | head -1)
        if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/epoch_2.pth" ] && grep -q "epoch: 2" "$RUN_DIR/log.txt" 2>/dev/null; then
            STATUS[S2a_short_train]=PASS; S2A_OK=1
        else
            STATUS[S2a_short_train]=FAIL; echo "    see $SMOKE/s2a_train.log"
        fi
    fi
else
    STATUS[S2a_short_train]=SKIP
fi

# S2b 单 batch 过拟合 200 步
if [ $S1_OK -eq 1 ]; then
    IMPL_FLAG=""
    [ "$MODE" = new ] && IMPL_FLAG="--impl new"
    run_step S2b_overfit $SMOKE/s2b_overfit.log $VENV $CODE/scripts/smoke_checks.py overfit \
        --train-root $TRAIN_ROOT --steps 200 $IMPL_FLAG
else
    STATUS[S2b_overfit]=SKIP
fi

# S3 ckpt 存取一致性（bevflow.models，两模式相同）
if [ "$MODE" = new ]; then
    CKPT=$RUN_DIR/ckpt/epoch_2.pth
else
    CKPT=$RUN_DIR/epoch_2.pth
fi
if [ $S2A_OK -eq 1 ] && [ -f $CKPT ]; then
    run_step S3 $SMOKE/s3.log $VENV $CODE/scripts/smoke_checks.py s3 --ckpt $CKPT
else
    STATUS[S3]=SKIP
fi

# S4 评测
if [ $S2A_OK -eq 1 ] && [ -f $CKPT ]; then
    if [ "$MODE" = new ]; then
        (
            cd $CODE && $VENV scripts/eval.py --ckpt $CKPT --data-root $VAL_ROOT \
                --split val --out-dir $SMOKE/eval --run-id smoke_${MODE}_$TS --bs 4
        ) > $SMOKE/s4_eval.log 2>&1
        if [ -f $SMOKE/eval/metrics.json ]; then
            STATUS[S4_eval]=PASS; $VENV -c "import json; d=json.load(open('$SMOKE/eval/metrics.json')); [print(f'    epe {k}: {v[\"mean\"]:.4f}') for k,v in d['epe'].items()]" 2>/dev/null
        else
            STATUS[S4_eval]=FAIL; echo "    see $SMOKE/s4_eval.log"
        fi
    else
        (
            cd $UPSTREAM &&
            $VENV eval.py -d $VAL_ROOT -m $CKPT -l $SMOKE/eval -s val -b 4 -w 2
        ) > $SMOKE/s4_eval.log 2>&1
        if [ -f $SMOKE/eval/eval_displacement.txt ]; then
            if $VENV $CODE/scripts/eval_parse.py --eval-file $SMOKE/eval/eval_displacement.txt \
                    --out $SMOKE/eval/metrics.json --run-id smoke_$TS --ckpt $CKPT \
                    --n-samples $(ls $VAL_ROOT | wc -l) \
                    > $SMOKE/s4_parse.log 2>&1; then
                STATUS[S4_eval]=PASS; tail -8 $SMOKE/s4_parse.log | sed 's/^/    /'
            else
                STATUS[S4_eval]=FAIL; echo "    parse failed, see $SMOKE/s4_parse.log"
            fi
        else
            STATUS[S4_eval]=FAIL; echo "    see $SMOKE/s4_eval.log"
        fi
    fi
else
    STATUS[S4_eval]=SKIP
fi

# S5 可视化 + 数值检查
if [ $S2A_OK -eq 1 ] && [ -f $CKPT ]; then
    if [ "$MODE" = new ]; then
        (
            cd $CODE && $VENV scripts/viz.py --val-root $VAL_ROOT --ckpt $CKPT \
                --out-dir $SMOKE/viz
        ) > $SMOKE/s5_viz.log 2>&1
        if [ -f $SMOKE/viz/viz_manifest.json ]; then STATUS[S5_viz]=PASS; else STATUS[S5_viz]=FAIL; echo "    see $SMOKE/s5_viz.log"; fi
    else
        if run_step S5_viz $SMOKE/s5_viz.log $VENV $CODE/scripts/viz_smoke.py \
                --val-root $VAL_ROOT --ckpt $CKPT --out-dir $SMOKE/viz; then :; fi
    fi
    if [ "${STATUS[S5_viz]}" = PASS ]; then
        run_step S5_check $SMOKE/s5_check.log $VENV $CODE/scripts/check_viz.py \
            --manifest $SMOKE/viz/viz_manifest.json
    else
        STATUS[S5_check]=SKIP
    fi
else
    STATUS[S5_viz]=SKIP
    STATUS[S5_check]=SKIP
fi

# 资源实测（不作为通过条件）
if [ $S1_OK -eq 1 ]; then
    {
        $VENV $CODE/scripts/smoke_checks.py res --train-root $TRAIN_ROOT --bs 4 --steps 20
        $VENV $CODE/scripts/smoke_checks.py res --train-root $TRAIN_ROOT --bs 2 --steps 20
    } > $SMOKE/resources.txt 2>&1
    STATUS[RES_resources]="INFO"
    cat $SMOKE/resources.txt | sed 's/^/    /'
fi

echo
echo "===== SMOKE [$MODE] $TS ====="
for k in S1 S2a_short_train S2b_overfit S3 S4_eval S5_viz S5_check RES_resources; do
    v=${STATUS[$k]:-UNSET}
    printf "%-18s %s\n" "$k" "$v"
done
echo "products: $SMOKE"

FAILS=$(for v in "${STATUS[@]}"; do [ "$v" = FAIL ] && echo x; done | wc -l)
[ $FAILS -gt 0 ] && exit 1
exit 0
