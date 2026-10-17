"""可视化渲染。风格对齐上游 plots.py（类别色 quiver），附误差图与静态噪点图。"""
import json
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLOR_MAP = {0: "c", 1: "m", 2: "k", 3: "y", 4: "r"}
CAT_NAMES = {0: "bg", 1: "bus", 2: "ped", 3: "bike", 4: "other"}
BORDER_METER = 4
VOXEL_XY = 0.25
DISP_DISPLAY_THD = 0.4
STATIC_THD_PER_FRAME = 0.01  # 0.2 m/s * (1/20) s
FUTURE_FRAMES = 20


def _quiver_panel(ax, field, cat_map_int, k, title):
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


def visualize_samples(model, dataset, idxs, out_dir, seed=0, ckpt=""):
    os.makedirs(out_dir, exist_ok=True)
    device = next(model.parameters()).device
    manifest = {
        "style": "plots.py quiver, category colors c/m/k/y/r",
        "ckpt": ckpt, "seed": seed, "idxs": [int(i) for i in idxs],
        "future_row": FUTURE_FRAMES - 1,
        "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=CODE,
                              capture_output=True, text=True).stdout.strip(),
        "samples": [],
    }

    for i, idx in enumerate(idxs):
        padded, disp, valid, non_empty, cat, past, future, motion = dataset[idx]

        x = torch.from_numpy(np.asarray(padded)).unsqueeze(0).to(device)
        with torch.no_grad():
            disp_pred, cell_pred, _ = model(x)
        disp_pred = disp_pred.view(1, -1, 256, 256, 2).cpu().numpy()
        for c in range(1, disp_pred.shape[1]):
            disp_pred[:, c] += disp_pred[:, c - 1]
        pred = disp_pred[0, -1]
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
            _quiver_panel(axes[0], display_gt, gt_cat, k, "Ground-truth (1.0s)")
            _quiver_panel(axes[1], display_pred, pred_cat, k, "Prediction (1.0s)")
        fig.savefig(os.path.join(out_dir, f"{i:02d}_idx{idx}_quiver.png"), bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=192)
        im = ax.imshow(err, cmap="magma", vmin=0, interpolation="nearest")
        ax.set_title("|pred-gt| (1.0s)", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label="m")
        fig.savefig(os.path.join(out_dir, f"{i:02d}_idx{idx}_err.png"), bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=192)
        vmax = np.nanmax(static_noise)
        im = ax.imshow(static_noise, cmap="viridis", vmin=0,
                       vmax=vmax if np.isfinite(vmax) and vmax > 0 else 1, interpolation="nearest")
        ax.set_title("pred |flow| on GT-static cells", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label="m")
        fig.savefig(os.path.join(out_dir, f"{i:02d}_idx{idx}_staticerr.png"), bbox_inches="tight")
        plt.close(fig)

        manifest["samples"].append({
            "idx": int(idx),
            "quiver": f"{i:02d}_idx{idx}_quiver.png",
            "err": f"{i:02d}_idx{idx}_err.png",
            "staticerr": f"{i:02d}_idx{idx}_staticerr.png",
        })

    with open(os.path.join(out_dir, "viz_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest
