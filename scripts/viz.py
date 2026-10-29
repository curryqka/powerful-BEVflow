# SPDX-License-Identifier: Apache-2.0
"""可视化薄入口。用法：
python scripts/viz.py --val-root <val 目录> --ckpt <ckpt> --out-dir <目录> [--n 4 --seed 0]
风格与上游 plots.py 对齐（类别色 quiver），另附误差图与静态噪点图。
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
import torch  # noqa: E402

from bevflow.datasets.motionnet import DatasetSingleSeq  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from bevflow.evaluators.viz import visualize_samples
    from scripts.eval import load_model

    dataset = DatasetSingleSeq(
        dataset_root=args.val_root, split="val", future_frame_skip=0,
        voxel_size=(0.25, 0.25, 0.4),
        area_extents=np.array([[-32.0, 32.0], [-32.0, 32.0], [-3.0, 2.0]]),
        num_category=5,
    )
    rng = np.random.RandomState(args.seed)
    idxs = sorted(rng.choice(len(dataset), args.n, replace=False).tolist())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    visualize_samples(model, dataset, idxs, args.out_dir, seed=args.seed, ckpt=args.ckpt)
    print("viz done:", args.out_dir)


if __name__ == "__main__":
    main()
