# SPDX-License-Identifier: Apache-2.0
"""数据统计：每 split 随机抽 20 个样本查形状/数值，统计三档占比与动态格子比例。

结果写 $DATA/preprocessed/motionnet_v1/data_stats.json。用法：
python scripts/check_data.py [--n 20]
"""
import argparse
import json
import os
import random
import sys

import numpy as np

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)

DATA = os.environ.get("BEVFLOW_DATA", "/media/nas_users/kaneqian/data/bevflow")
ROOT = os.path.join(DATA, "preprocessed", "motionnet_v1")
STATIC_THD = 0.01      # 0.2 m/s × 1/20 s，每帧位移上限，与评测协议一致
SLOW_THD = 5.0         # m/s，末帧位移上限


def load_sample_train(root, seq_dir):
    files = sorted(os.listdir(os.path.join(root, seq_dir)))
    return [np.load(os.path.join(root, seq_dir, f), allow_pickle=True).item() for f in files]


def tier_stats(sample_list):
    """三档按评测协议：static=所有帧位移≤0.01m；slow/fast 按末帧位移（1s 对应 5 m/s 阈值）。"""
    n_static = n_slow = n_fast = n_valid = 0
    for dicts in sample_list:
        for d in dicts:
            dims = d["3d_dimension"]
            n_future = d["num_future_pcs"]
            px = d["pixel_indices"]
            disp = np.zeros((n_future, dims[0], dims[1], 2), dtype=np.float32)
            disp[:, px[:, 0], px[:, 1], :] = d["disp_field"]
            valid = np.zeros((n_future, dims[0], dims[1]), dtype=np.float32)
            valid[:, px[:, 0], px[:, 1]] = d["valid_pixel_map"].astype(np.float32)

            norm = np.linalg.norm(disp, ord=2, axis=-1)  # (F, h, w)
            non_empty = valid[-1] > 0
            static = (norm <= STATIC_THD).all(axis=0) & non_empty
            moving = non_empty & ~static
            last_norm = norm[-1]
            n_static += int(static.sum())
            n_slow += int((moving & (last_norm <= SLOW_THD)).sum())
            n_fast += int((moving & (last_norm > SLOW_THD)).sum())
            n_valid += int(non_empty.sum())
    return n_static, n_slow, n_fast, n_valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    rng = random.Random(0)
    stats = {"n_per_split": args.n, "splits": {}}

    for split, kind in (("train", "dirs"), ("val", "flat"), ("test", "flat")):
        root = os.path.join(ROOT, split)
        if kind == "dirs":
            names = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            picks = rng.sample(names, min(args.n, len(names)))
            samples = [load_sample_train(root, d) for d in picks]
        else:
            names = [f for f in os.listdir(root) if f.endswith(".npy")]
            picks = rng.sample(names, min(args.n, len(names)))
            samples = [[np.load(os.path.join(root, f), allow_pickle=True).item()] for f in picks]

        bad = 0
        for dicts in samples:
            for d in dicts:
                dims = tuple(d["3d_dimension"])
                assert dims == (256, 256, 13), f"unexpected dims {dims}"
                for k in ("disp_field", "valid_pixel_map", "pixel_cat_map", "pixel_instance_ids"):
                    assert np.isfinite(np.asarray(d[k], dtype=np.float32)).all(), f"non-finite {k}"

        n_static, n_slow, n_fast, n_valid = tier_stats(samples)
        stats["splits"][split] = {
            "checked": picks[:5] + ["..."],
            "finite_ok": True,
            "cells": {"static": n_static, "slow": n_slow, "fast": n_fast, "valid": n_valid},
            "ratio": {
                "static": round(n_static / max(n_valid, 1), 4),
                "slow": round(n_slow / max(n_valid, 1), 4),
                "fast": round(n_fast / max(n_valid, 1), 4),
                "dynamic_of_valid": round((n_slow + n_fast) / max(n_valid, 1), 4),
            },
        }
        print(split, stats["splits"][split]["ratio"])

    out = os.path.join(ROOT, "data_stats.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=1)
    print("written:", out)


if __name__ == "__main__":
    main()
