RUN_ROOT="/SHARE/qiankangan/bevflow"
cd /workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion
pip install -r requirements.txt
PORT=$1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --use_env --nproc_per_node=4 --master_port=$PORT train_multigpu.py \
--data $RUN_ROOT/input-data --evaldata $RUN_ROOT/val \
--batch 1 --nepoch 45 --nworker 4 --lr 0.0008 \
--use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 15.0 --use_dw \
--log --model GridInsMotionNet --mask \
--logpath $RUN_ROOT/logs/multigpu_GridInsMotionNet
# --logpath $RUN_ROOT/logs/test_multigpu_mask_fn005_lr0016 --mask
