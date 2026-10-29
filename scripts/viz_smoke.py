# SPDX-License-Identifier: Apache-2.0
"""冒烟可视化：固定 4 个 val 样本，风格对齐上游 plots.py。

每个样本四张 PNG：GT quiver，预测 quiver（类别着色，同 plots.py），|pred-gt| 误差图，
GT 静态格子上的预测幅值（静态噪点图）。后两张是 plots.py 没有的补充面板。
"""
import argparse
import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(CODE, "third_party", "MotionNet")
sys.path.insert(0, UPSTREAM)
sys.path.insert(0, os.path.join(UPSTREAM, "nuscenes-devkit", "python-sdk"))

HEIGHT_FEAT = 13
FUTURE_FRAMES = 20
STATIC_THD_PER_FRAME = 0.01  # 0.2 m/s * (1/20) s，与 eval.py 一致

# 与 plots.py 一致
COLOR_MAP = {0: "c", 1: "m", 2: "k", 3: "y", 4: "r"}
CAT_NAMES = {0: "bg", 1: "bus", 2: "ped", 3: "bike", 4: "other"}
BORDER_METER = 4
VOXEL_XY = 0.25
DISP_DISPLAY_THD = 0.4  # plots.py：小于该幅值（米）的位移箭头归零显示


def quiver_panel(ax, field, cat_map_int, k, title):
    idx = np.arange(field.shape[0])
    idx_x, idx_y = np.meshgrid(idx, idx, indexing="ij")
    mask = cat_map_int == (k + 1)
    q = ax.quiver(
        idx_x[mask], idx_y[mask],
        field[..., 0][mask] / VOXEL_XY, field[..., 1][mask] / VOXEL_XY,
        angles="xy", scale_units="xy", scale=1, color=COLOR_MAP[k],
    )
    ax.quiverkey(q, X=0.0 + k / 5.0, Y=1.1, U=20, label=CAT_NAMES[k], labelpos="E")
    b = int(BORDER_METER / VOXEL_XY)
    ax.set_xlim(b, field.shape[0] - b)
    ax.set_ylim(b, field.shape[1] - b)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sys.argv = ["train_multi_seq.py", "-d", "x", "--batch", "1", "--nepoch", "1", "--nworker", "0"]
    from data.nuscenes_dataloader import DatasetSingleSeq
    from model import MotionNet

    dataset = DatasetSingleSeq(
        dataset_root=args.val_root, split="val", future_frame_skip=0,
        voxel_size=(0.25, 0.25, 0.4),
        area_extents=np.array([[-32.0, 32.0], [-32.0, 32.0], [-3.0, 2.0]]),
        num_category=5,
    )
    rng = np.random.RandomState(args.seed)
    idxs = sorted(rng.choice(len(dataset), args.n, replace=False).tolist())

    device = torch.device("cuda")
    model = torch.nn.DataParallel(MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT))
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
    model = model.to(device).eval()

    manifest = {"style": "plots.py quiver, category colors c/m/k/y/r", "val_root": args.val_root,
                "ckpt": args.ckpt, "seed": args.seed, "idxs": idxs, "future_row": FUTURE_FRAMES - 1,
                "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE,
                                      capture_output=True, text=True).stdout.strip(),
                "samples": []}

    for i, idx in enumerate(idxs):
        padded, disp, valid, non_empty, cat, past, future, motion = dataset[idx]

        x = torch.from_numpy(padded).unsqueeze(0).to(device)
        with torch.no_grad():
            disp_pred, cell_pred, _ = model(x)
        # 模型原始输出 (B*F, 2, h, w)：与 plots.py 相同的解析，勿改成 channel-last view
        disp_pred = disp_pred.view(1, -1, 2, 256, 256)[0].cpu().numpy()   # (F, 2, h, w)
        disp_pred = np.transpose(disp_pred, (0, 2, 3, 1))                 # (F, h, w, 2)
        for c in range(1, disp_pred.shape[0]):
            disp_pred[c] += disp_pred[c - 1]
        pred = disp_pred[-1]
        cat_pred = torch.argmax(cell_pred[0], dim=0).cpu().numpy() + 1

        gt = disp[-1]
        one_hot = cat.max(-1) == 1.0
        gt_cat = (np.argmax(cat, -1) + 1) * non_empty * one_hot
        pred_cat = cat_pred * non_empty * one_hot

        display_gt = gt.copy()
        display_gt[np.linalg.norm(gt, axis=-1) <= DISP_DISPLAY_THD] = 0
        display_pred = pred.copy()
        display_pred[np.linalg.norm(pred, axis=-1) <= DISP_DISPLAY_THD] = 0

        err = np.linalg.norm(pred - gt, axis=-1)
        err[non_empty == 0] = np.nan
        gt_norm_all = np.linalg.norm(disp, axis=-1)
        static = (gt_norm_all <= STATIC_THD_PER_FRAME).all(axis=0) & (non_empty > 0)
        static_noise = np.where(static, np.linalg.norm(pred, axis=-1), np.nan)

        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), dpi=192)
        for k in range(5):
            quiver_panel(axes[0], display_gt, gt_cat, k, "Ground-truth (1.0s)")
            quiver_panel(axes[1], display_pred, pred_cat, k, "Prediction (1.0s)")
        fig.savefig(os.path.join(args.out_dir, f"{i:02d}_idx{idx}_quiver.png"), bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=192)
        im = ax.imshow(err, cmap="magma", vmin=0, interpolation="nearest")
        ax.set_title("|pred-gt| (1.0s)", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label="m")
        fig.savefig(os.path.join(args.out_dir, f"{i:02d}_idx{idx}_err.png"), bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=192)
        vmax = np.nanmax(static_noise)
        im = ax.imshow(static_noise, cmap="viridis", vmin=0,
                       vmax=vmax if np.isfinite(vmax) and vmax > 0 else 1, interpolation="nearest")
        ax.set_title("pred |flow| on GT-static cells", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label="m")
        fig.savefig(os.path.join(args.out_dir, f"{i:02d}_idx{idx}_staticerr.png"), bbox_inches="tight")
        plt.close(fig)

        manifest["samples"].append({
            "idx": int(idx),
            "quiver": f"{i:02d}_idx{idx}_quiver.png",
            "err": f"{i:02d}_idx{idx}_err.png",
            "staticerr": f"{i:02d}_idx{idx}_staticerr.png",
        })
        print(f"sample idx={idx} done, static_cells={int(static.sum())}")

    with open(os.path.join(args.out_dir, "viz_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("viz done:", args.out_dir)


if __name__ == "__main__":
    main()
