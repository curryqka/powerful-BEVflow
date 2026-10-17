"""数据生产薄入口。用法：
python scripts/gen_data.py --split train [--scenes 116 229] \
    --savepath $DATA/preprocessed/motionnet_v1/train
split 文件默认取 $DATA/splits/split.npy；--scenes 给场景下标子集（冒烟/分片）。
"""
import argparse
import os
import sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(CODE, "third_party", "MotionNet")
for p in (CODE, UPSTREAM, os.path.join(UPSTREAM, "nuscenes-devkit", "python-sdk")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

DATA = os.environ.get("BEVFLOW_DATA", "/media/nas_users/kaneqian/data/bevflow")
RAW = os.environ.get("RAW_NUSCENES", "/media/nas_users/kaneqian/data/Nuscences")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--savepath", required=True)
    parser.add_argument("--scenes", type=int, nargs="*", default=None, help="场景下标子集")
    parser.add_argument("--split-file", default=os.path.join(DATA, "splits", "split.npy"))
    args = parser.parse_args()

    from bevflow.preprocess.generate import generate_scenes, load_nuscenes

    scenes = np.load(args.split_file, allow_pickle=True).item()[args.split]
    scene_ids = sorted(int(s.split("_")[1]) for s in scenes)
    if args.scenes is not None:
        scene_ids = [s for s in scene_ids if s in set(args.scenes)]
    assert scene_ids, "no scenes selected"

    nusc = load_nuscenes(RAW)
    n = generate_scenes(nusc, scene_ids, args.split, args.savepath, log_every=20)
    print(f"done: {n} seq groups into {args.savepath}")


if __name__ == "__main__":
    main()
