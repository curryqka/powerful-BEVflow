"""从 train split 固定种子抽 10 个场景做冒烟：6/2/2。"""
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--split-file", default="/media/nas_users/kaneqian/data/bevflow/splits/split.npy")
parser.add_argument("--out", default="/media/nas_users/kaneqian/data/bevflow/splits/mini_split.npy")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

s = np.load(args.split_file, allow_pickle=True).item()
train_ids = sorted(s["train"])
rng = np.random.RandomState(args.seed)
pick = sorted(rng.choice(len(train_ids), 10, replace=False))
scenes = [train_ids[i] for i in pick]
mini = {"train": scenes[:6], "val": scenes[6:8], "test": scenes[8:]}
assert len(mini["train"]) == 6 and len(mini["val"]) == 2 and len(mini["test"]) == 2
np.save(args.out, mini)
print("mini split:", {k: len(v) for k, v in mini.items()}, mini)
