"""冒烟检查子命令：s1 数据检查，overfit 单 batch 过拟合，s3 ckpt 一致性，res 资源实测。

用法：python scripts/smoke_checks.py <s1|overfit|s3|res> [参数]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(CODE, "third_party", "MotionNet")
sys.path.insert(0, CODE)
sys.path.insert(0, UPSTREAM)
sys.path.insert(0, os.path.join(UPSTREAM, "nuscenes-devkit", "python-sdk"))

HEIGHT_FEAT = 13
FUTURE_FRAMES = 20
VOXEL_SIZE = (0.25, 0.25, 0.4)
AREA_EXTENTS = np.array([[-32.0, 32.0], [-32.0, 32.0], [-3.0, 2.0]])


def _load_train_dataset(root):
    from data.nuscenes_dataloader import TrainDatasetMultiSeq

    return TrainDatasetMultiSeq(
        dataset_root=root,
        future_frame_skip=0,
        voxel_size=VOXEL_SIZE,
        area_extents=AREA_EXTENTS,
        num_past_frames=5,
        num_future_frames=FUTURE_FRAMES,
        num_category=5,
    )


def _load_single_seq_dataset(root, split):
    from data.nuscenes_dataloader import DatasetSingleSeq

    return DatasetSingleSeq(
        dataset_root=root,
        split=split,
        future_frame_skip=0,
        voxel_size=VOXEL_SIZE,
        area_extents=AREA_EXTENTS,
        num_category=5,
    )


def cmd_s1(args):
    for root in (args.train_root, args.val_root):
        assert os.path.isdir(root), f"missing data root: {root}"

    trainset = _load_train_dataset(args.train_root)
    assert len(trainset) > 0
    for idx in range(min(20, len(trainset))):
        padded, disp, valid, non_empty, cat, trans, motion, inst, npf, nfu = trainset[idx]
        n_adj = padded.shape[0]
        assert padded.shape == (n_adj, 5, 256, 256, HEIGHT_FEAT), f"voxel shape {padded.shape}"
        assert disp.shape == (n_adj, FUTURE_FRAMES, 256, 256, 2), f"disp shape {disp.shape}"
        assert cat.shape[-1] == 5
        for t in (padded, disp, valid, non_empty, cat, motion):
            assert np.isfinite(t).all(), f"non-finite at train idx {idx}"

    valset = _load_single_seq_dataset(args.val_root, "val")
    assert len(valset) > 0
    for idx in range(min(20, len(valset))):
        padded, disp, valid, non_empty, cat, past, future, motion = valset[idx]
        assert padded.shape == (5, 256, 256, HEIGHT_FEAT), f"val voxel shape {padded.shape}"
        assert disp.shape == (FUTURE_FRAMES, 256, 256, 2), f"val disp shape {disp.shape}"
        for t in (padded, disp, valid, non_empty, cat):
            assert np.isfinite(t).all(), f"non-finite at val idx {idx}"

    print(f"S1 PASS train_seqs={len(trainset)} val_seqs={len(valset)}")


def cmd_overfit(args):
    dataset = _load_train_dataset(args.train_root)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    device = torch.device("cuda")
    (padded, disp, valid, non_empty, cat, trans, motion_gt, inst, npf, nfu) = batch
    padded = padded.view(-1, npf[0].item(), 256, 256, HEIGHT_FEAT).to(device)

    if args.impl == "new":
        from bevflow.models.losses import LossConfig, compute_losses
        from bevflow.models.motionnet import MotionNet

        model = torch.nn.DataParallel(
            MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT)
        ).to(device).train()
        loss_cfg = LossConfig()

        def one_step(optimizer, criterion):
            disp_pred, class_pred, motion_pred = model(padded)
            total, parts = compute_losses(
                loss_cfg, disp, valid, cat, disp_pred, non_empty, class_pred,
                motion_gt, motion_pred, trans, inst, device,
            )
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            return parts
    else:
        sys.argv = ["train_multi_seq.py", "-d", args.train_root, "--batch", "1", "--nepoch", "1", "--nworker", "0"]
        import train_multi_seq as tm

        model = torch.nn.DataParallel(
            tm.MotionNet(out_seq_len=tm.out_seq_len, motion_category_num=2, height_feat_size=tm.height_feat_size)
        ).to(device).train()

        def one_step(optimizer, criterion):
            disp_pred, class_pred, motion_pred = model(padded)
            out = tm.compute_and_bp_loss(
                optimizer, device, nfu[0].item(),
                disp, valid, cat, disp_pred, criterion, non_empty, class_pred,
                motion_gt, motion_pred, trans, inst,
            )
            return out

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.SmoothL1Loss(reduction="none")

    losses = []
    for step in range(args.steps):
        parts = one_step(optimizer, criterion)
        assert parts[0] is not None, "no valid pixels in the fixed batch"
        losses.append(float(parts[0]))
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step}: loss_disp={losses[-1]:.6f}")

    first10 = float(np.mean(losses[:10]))
    last10 = float(np.mean(losses[-10:]))
    print(f"first10={first10:.6f} last10={last10:.6f} ratio={last10 / first10:.4f}")
    assert np.isfinite(losses).all()
    assert last10 < first10 * 0.5, f"loss did not halve: {first10} -> {last10}"
    print("OVERFIT PASS")


def cmd_s3(args):
    from bevflow.models.motionnet import MotionNet

    device = torch.device("cuda")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = torch.nn.DataParallel(MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 5, 256, 256, HEIGHT_FEAT, generator=g).to(device)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    max_diff = max((a - b).abs().max().item() for a, b in zip(y1, y2))
    print(f"max abs diff over two forwards: {max_diff:.3e}")
    assert max_diff < 1e-6, f"ckpt forward not deterministic: {max_diff}"
    print("S3 PASS")


def cmd_res(args):
    from bevflow.models.losses import LossConfig, compute_losses
    from bevflow.models.motionnet import MotionNet

    dataset = _load_train_dataset(args.train_root)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.bs, shuffle=True, num_workers=2, pin_memory=True
    )
    device = torch.device("cuda")
    model = torch.nn.DataParallel(
        MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT)
    ).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.SmoothL1Loss(reduction="none")
    loss_cfg = LossConfig()

    torch.cuda.reset_peak_memory_stats()
    it = iter(loader)
    n, t_total, t_compute = 0, 0.0, 0.0
    while n < args.steps:
        t0 = time.perf_counter()
        batch = next(it)
        t_fetch = time.perf_counter()
        (padded, disp, valid, non_empty, cat, trans, motion_gt, inst, npf, nfu) = batch
        padded = padded.view(-1, npf[0].item(), 256, 256, HEIGHT_FEAT).to(device)
        disp_pred, class_pred, motion_pred = model(padded)
        total, parts = compute_losses(
            loss_cfg, disp, valid, cat, disp_pred, non_empty, class_pred,
            motion_gt, motion_pred, trans, inst, device,
        )
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        t_total += t1 - t0
        t_compute += t1 - t_fetch
        n += 1

    peak = torch.cuda.max_memory_allocated() / 2**30
    e2e_ms = t_total / n * 1000
    compute_ms = t_compute / n * 1000
    data_share = (1 - t_compute / t_total) * 100
    print(f"bs={args.bs} steps={n} peak_mem={peak:.2f}GiB e2e_ms={e2e_ms:.0f} "
          f"compute_ms={compute_ms:.0f} data_share={data_share:.0f}%")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("s1")
    s1.add_argument("--train-root", required=True)
    s1.add_argument("--val-root", required=True)

    of = sub.add_parser("overfit")
    of.add_argument("--train-root", required=True)
    of.add_argument("--steps", type=int, default=200)
    of.add_argument("--impl", default="upstream", choices=["upstream", "new"])

    s3 = sub.add_parser("s3")
    s3.add_argument("--ckpt", required=True)

    res = sub.add_parser("res")
    res.add_argument("--train-root", required=True)
    res.add_argument("--bs", type=int, default=4)
    res.add_argument("--steps", type=int, default=20)

    args = p.parse_args()
    {"s1": cmd_s1, "overfit": cmd_overfit, "s3": cmd_s3, "res": cmd_res}[args.cmd](args)


if __name__ == "__main__":
    main()
