#!/bin/bash
# 训练自动恢复看守：每 120 秒检查各 run；进程不在、epoch 未满 → 自动 resume。
# 规格文件每行：run_id|GPU|config|set_args|total_epochs
# 启动：setsid nohup bash scripts/watchdog.sh &
set -u
CODE=/home/kaneqian/BEVflow
EXP=/media/nas_users/kaneqian/Experiment/bevflow
VENV=$EXP/_env/bevflow/bin/python
SPECS=${WATCHDOG_SPECS:-$EXP/_cache/watchdog_runs.txt}
LOG=$EXP/_cache/cmdlogs/watchdog.log
export BEVFLOW_DATA=/media/nas_users/kaneqian/data/bevflow

echo "$(date) watchdog start pid $$ specs=$SPECS" >> $LOG
while true; do
    while IFS='|' read -r rid gpu cfg setargs total; do
        [ -z "$rid" ] && continue
        dir=$(ls -d $EXP/*/"$rid" 2>/dev/null | head -1)
        [ -z "$dir" ] && continue

        last_epoch=$($VENV -c "
import json
try:
    epochs = [json.loads(l)['epoch'] for l in open('$dir/logs/train.jsonl')
              if '\"type\": \"epoch\"' in l]
    print(epochs[-1] if epochs else 0)
except FileNotFoundError:
    print(0)" 2>/dev/null)
        [ "${last_epoch:-0}" -ge "$total" ] && continue

        if pgrep -f "run_id=$rid" > /dev/null; then continue; fi

        echo "$(date) watchdog: $rid 无进程，epoch $last_epoch/$total，GPU$gpu 重启" >> $LOG
        cd $CODE
        setsid nohup env CUDA_VISIBLE_DEVICES=$gpu $VENV scripts/train.py \
            --config "configs/exp/$cfg" --set run.run_id=$rid $setargs \
            >> $EXP/_cache/cmdlogs/watchdog_$rid.log 2>&1 &
        sleep 30
    done < "$SPECS"
    sleep 120
done
