export PYTHONPATH=/home/qiankangan/MotionNet:$PYTHONPATH
export PYTHONPATH=/home/qiankangan/MotionNet/nuscenes-devkit/python-sdk:$PYTHONPATH
OMP_NUM_THREADS=1 python train_multi_seq.py --data /mnt/personal/qiankangan/at128_0/train --evaldata /mnt/personal/qiankangan/at128_0/train \
--batch 32 --nepoch 100 --nworker 2  --log --logpath /home/qiankangan/logs/at128_0v1 --gpu "0" \
--use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 


OMP_NUM_THREADS=1 python train_multi_seq_wt128.py --data /mnt/personal/qiankangan/wt128_0/train --evaldata /mnt/personal/qiankangan/wt128_0/train \
--batch 32 --nepoch 100 --nworker 2  --log --logpath /home/qiankangan/logs/wt128_0v1 --gpu "5" \
--use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 


OMP_NUM_THREADS=1 python train_multi_seq.py --data /mnt/personal/qiankangan/input-data --evaldata /mnt/personal/qiankangan/val \
--batch 32 --nepoch 45 --nworker 2  --log --logpath /home/qiankangan/logs/nuscenes --gpu "3" \
--use_bg_tc --reg_weight_bg_tc 0.1 --use_fg_tc --reg_weight_fg_tc 2.5 --use_sc --reg_weight_sc 2.0 

OMP_NUM_THREADS=1 python eval.py --data /mnt/personal/qiankangan/at128_0/train \
--model /home/qiankangan/MotionNet/logs/train_multi_seq/2024-07-31_11-44-04/epoch_300.pth \
--split test --log logs/hetest --bs 1 --net MotionNet

OMP_NUM_THREADS=1 python eval.py --data /mnt/personal/qiankangan/val \
--model /home/qiankangan/MotionNet/logs/train_multi_seq/2024-07-31_11-44-04/epoch_300.pth \
--split test --log logs/hetest2 --bs 1 --net MotionNet


OMP_NUM_THREADS=1 python eval.py --data /mnt/personal/qiankangan/val \
--model /home/qiankangan/MotionNet/logs/train_multi_seq/2024-07-31_11-03-56/epoch_40.pth \
--split test --log logs/nutest --bs 1 --net MotionNet

OMP_NUM_THREADS=1 python eval.py --data /mnt/personal/qiankangan/at128_0/train \
--model /home/qiankangan/MotionNet/logs/train_multi_seq/2024-07-31_11-03-56/epoch_40.pth \
--split test --log logs/nutest2 --bs 1 --net MotionNet
