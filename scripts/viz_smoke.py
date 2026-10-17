"""冒烟可视化：固定 4 个 val 样本，各出 GT / 预测 / 误差 / 静态误差 四张 PNG。

颜色编码：HSV 流场图，角度=色相，幅值/上限=明度，固定色轮与色标。
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
from matplotlib.colors import hsv_to_rgb

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(CODE, "third_party", "MotionNet")
sys.path.insert(0, UPSTREAM)
sys.path.insert(0, os.path.join(UPSTREAM, "nuscenes-devkit", "python-sdk"))

HEIGHT_FEAT = 13
FUTURE_FRAMES = 20
STATIC_THD_PER_FRAME = 0.01  # 0.2 m/s * (1/20) s，与 eval.py 一致


def flow_to_rgb(flow, max_mag, valid):
    mag = np.linalg.norm(flow, axis=-1)
    ang = np.arctan2(flow[..., 1], flow[..., 0])
    hue = (ang + np.pi) / (2 * np.pi)
    val = 0.15 + 0.85 * np.clip(mag / max_mag, 0, 1)
    rgb = hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.9), val], axis=-1))
    rgb[~valid] = 0.92
    return rgb


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

    gts, preds = [], []
    for idx in idxs:
        padded, disp, valid, non_empty, cat, past, future, motion = dataset[idx]
        x = torch.from_numpy(padded).unsqueeze(0).to(device)
        with torch.no_grad():
            disp_pred, _, _ = model(x)
        disp_pred = disp_pred.view(1, -1, 256, 256, 2).cpu().numpy()
        for c in range(1, disp_pred.shape[1]):
            disp_pred[:, c] += disp_pred[:, c - 1]
        gt = disp[-1]          # (256,256,2) 第 20 帧 = 1.0s
        pred = disp_pred[0, -1]
        gts.append((gt, non_empty, disp))
        preds.append(pred)

    max_mag = max(float(np.linalg.norm(g, axis=-1).max()) for g, _, _ in gts)
    max_mag = float(np.ceil(max_mag * 2) / 2)
    max_err = max(float(np.linalg.norm(p - g, axis=-1).max()) for (g, _, _), p in zip(gts, preds))

    manifest = {"val_root": args.val_root, "ckpt": args.ckpt, "seed": args.seed,
                "idxs": idxs, "future_row": FUTURE_FRAMES - 1, "max_mag_gt": max_mag,
                "max_err": max_err, "git": subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=CODE, capture_output=True, text=True
                ).stdout.strip(),
                "samples": []}

    for i, (idx, (gt, non_empty, disp_all), pred) in enumerate(zip(idxs, gts, preds)):
        valid = non_empty > 0
        err = np.linalg.norm(pred - gt, axis=-1)
        gt_norm_all = np.linalg.norm(disp_all, axis=-1)      # (20,256,256)
        static = (gt_norm_all <= STATIC_THD_PER_FRAME).all(axis=0) & valid

        panels = [
            ("gt", flow_to_rgb(gt, max_mag, valid), "GT flow (1.0s)"),
            ("pred", flow_to_rgb(pred, max_mag, valid), "Pred flow (1.0s)"),
        ]
        for tag, img, title in panels:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=192)
            ax.imshow(img, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            sm = plt.cm.ScalarMappable(matplotlib.colors.Normalize(0, max_mag), plt.cm.hsv)
            fig.colorbar(sm, ax=ax, fraction=0.046, label="|flow| m")
            f = os.path.join(args.out_dir, f"{i:02d}_idx{idx}_{tag}.png")
            fig.savefig(f, bbox_inches="tight")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(4, 4), dpi=192)
        im = ax.imshow(err, cmap="magma", vmin=0, vmax=max_err, interpolation="nearest")
        ax.set_title("|pred-gt| (1.0s)", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label="m")
        f = os.path.join(args.out_dir, f"{i:02d}_idx{idx}_err.png")
        fig.savefig(f, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4, 4), dpi=192)
        pred_mag = np.linalg.norm(pred, axis=-1)
        m = np.where(static, pred_mag, np.nan)
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=np.nanmax(m) if np.isfinite(np.nanmax(m)) else 1,
                       interpolation="nearest")
        ax.set_title("pred |flow| on GT-static cells", fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label="m")
        f = os.path.join(args.out_dir, f"{i:02d}_idx{idx}_staticerr.png")
        fig.savefig(f, bbox_inches="tight")
        plt.close(fig)

        manifest["samples"].append({
            "idx": int(idx),
            "gt": f"{i:02d}_idx{idx}_gt.png",
            "pred": f"{i:02d}_idx{idx}_pred.png",
            "err": f"{i:02d}_idx{idx}_err.png",
            "staticerr": f"{i:02d}_idx{idx}_staticerr.png",
        })
        print(f"sample idx={idx} done, static_cells={int(static.sum())}")

    with open(os.path.join(args.out_dir, "viz_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("viz done:", args.out_dir)


if __name__ == "__main__":
    main()
