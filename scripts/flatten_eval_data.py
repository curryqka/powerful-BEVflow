"""把 gen_data 输出的 <scene>_<seq>/N.npy 拍平成 <scene>_<seq>_<N>.npy。

上游 DatasetSingleSeq（eval.py 用）只认扁平目录下的 .npy 文件，见
third_party/MotionNet/data/nuscenes_dataloader.py 的 seq_files 构造。
"""
import argparse
import os
import shutil

p = argparse.ArgumentParser()
p.add_argument("root", help="eval 数据根目录，如 $DATA/preprocessed/motionnet_v1/val")
args = p.parse_args()

moved = 0
for d in sorted(os.listdir(args.root)):
    subdir = os.path.join(args.root, d)
    if not os.path.isdir(subdir):
        continue
    for f in sorted(os.listdir(subdir)):
        assert f.endswith(".npy"), f"unexpected file {f} in {subdir}"
        src = os.path.join(subdir, f)
        dst = os.path.join(args.root, f"{d}_{f}")
        assert not os.path.exists(dst), f"target exists: {dst}"
        shutil.move(src, dst)
        moved += 1
    os.rmdir(subdir)
print(f"flattened {moved} files under {args.root}")
