"""多序列训练器。训练循环从上游 train_multi_seq.py 搬运，数值行为一致：
- Adam lr=0.0016，MultiStepLR milestones [10,20,30,40] gamma 0.5
- scheduler.step() 在 epoch 开始时调用（上游固有顺序，保持）
- ckpt 保存策略：epoch 1、每 5 个、以及 >20 的每个 epoch（覆盖"最后 10 个"需求）

新增（均为 experiment-run skill 规定的机制，不影响数值）：
- train.jsonl 步级/轮级日志，--resume auto，固定种子，run_manifest.json，tensorboard 镜像。
"""
import datetime
import hashlib
import json
import os
import socket
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bevflow.models.losses import LossConfig, compute_losses
from bevflow.models.motionnet import MotionNet
from bevflow.utils.seed import fix_all_seeds

HEIGHT_FEAT = 13


def _git_info(code_dir):
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_dir, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=code_dir, capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "diff"], cwd=code_dir, capture_output=True, text=True).stdout
    return sha, bool(dirty), diff


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_latest_ckpt(ckpt_dir):
    """resume auto：返回最新 epoch_N.pth 的路径与 N，没有则 (None, 0)。"""
    best, best_n = None, 0
    if os.path.isdir(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.startswith("epoch_") and f.endswith(".pth"):
                try:
                    n = int(f[6:-4])
                except ValueError:
                    continue
                if n > best_n:
                    best, best_n = os.path.join(ckpt_dir, f), n
    return best, best_n


def train(cfg, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(os.path.join(logs_dir, "tb"), exist_ok=True)

    # -- 种子与配置快照
    seed = cfg["train"].get("seed", 0)
    fix_all_seeds(seed)
    with open(os.path.join(run_dir, "config_snapshot.yaml"), "w") as f:
        json.dump(cfg, f, indent=1, default=str)

    # -- run manifest
    code_dir = cfg["_code_dir"]
    git_sha, git_dirty, git_diff = _git_info(code_dir)
    if git_dirty:
        with open(os.path.join(run_dir, "code.diff"), "w") as f:
            f.write(git_diff)
    manifest = {
        "run_id": cfg["run"]["run_id"],
        "git": git_sha,
        "git_dirty": git_dirty,
        "config_sha": hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "data_card_sha": None,
        "seed": seed,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "start": datetime.datetime.now().isoformat(),
        "cmd": " ".join(sys_argv()),
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    from scripts.registry import append as _registry_append

    _registry_append({
        "run_id": manifest["run_id"], "exp_id": cfg["run"].get("exp_id", ""), "seed": seed,
        "status": "running", "config": cfg.get("_config_path", ""),
        "exp_path": run_dir,
    })

    # -- 数据
    from bevflow.datasets.motionnet import AREA_EXTENTS, VOXEL_SIZE, TrainDatasetMultiSeq

    card_path = os.path.join(os.path.dirname(cfg["data"]["train_root"]), "data_card.json")
    if os.path.exists(card_path):
        with open(card_path) as f:
            card = json.load(f)
        p = card["params"]
        assert list(p["voxel_size"]) == list(VOXEL_SIZE), f"data card voxel mismatch: {p['voxel_size']}"
        assert p["height_bins"] == HEIGHT_FEAT and p["past_frames"] == 5 and p["future_frames"] == 20
        assert p["range_xy"] == [AREA_EXTENTS[0][0], AREA_EXTENTS[0][1]], "data card range mismatch"

    trainset = TrainDatasetMultiSeq(
        dataset_root=cfg["data"]["train_root"],
        future_frame_skip=0,
        num_past_frames=5,
        num_future_frames=20,
        num_category=5,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"].get("workers", 4),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- 模型与优化器（超参 = 上游默认）
    model = nn.DataParallel(MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT))
    model = model.to(device)
    criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=cfg["optim"].get("lr", 0.0016))
    milestones = cfg["optim"].get("milestones", [10, 20, 30, 40])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    start_epoch = 1
    resume = cfg["train"].get("resume", "auto")
    latest, latest_n = find_latest_ckpt(ckpt_dir)
    latest_pt = os.path.join(ckpt_dir, "latest.pt")
    if resume == "auto" and os.path.exists(latest_pt):
        # 实例停止保险：每 500 步滚动保存；恢复时状态续接，epoch 数据从头重过（随机顺序无害）
        checkpoint = torch.load(latest_pt, map_location="cpu", weights_only=False)
        if latest is None or checkpoint["epoch"] > latest_n:
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"resumed mid-epoch state from {latest_pt} at epoch {checkpoint['epoch']}")
    elif resume == "auto" and latest is not None:
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"resumed from {latest} at epoch {latest_n}")

    loss_cfg = LossConfig(**cfg.get("loss", {}))

    # -- 日志
    train_jsonl = os.path.join(logs_dir, "train.jsonl")
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(os.path.join(logs_dir, "tb"))
    except ImportError:
        tb = None
    log_interval = cfg["log"].get("interval", 50)
    val_interval = cfg["train"].get("val_interval", 5)
    num_epochs = cfg["train"]["epochs"]

    if resume == "auto" and latest is not None:
        with open(train_jsonl, "a") as f:
            f.write(json.dumps({"type": "resume", "step_epoch": latest_n}) + "\n")

    def write_jsonl(record):
        with open(train_jsonl, "a") as f:
            f.write(json.dumps(record) + "\n")

    from bevflow.evaluators.motion_eval import evaluate

    t_start = time.time()
    for epoch in range(start_epoch, num_epochs + 1):
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()  # 上游顺序：epoch 开始时 step
        model.train()
        epoch_loss, n_steps = 0.0, 0
        t_epoch = time.time()

        for i, data in enumerate(trainloader, 0):
            (
                padded, disp_gt, valid, non_empty, cat, trans, motion_gt, inst, npf, nfu
            ) = data
            padded = padded.view(-1, npf[0].item(), 256, 256, HEIGHT_FEAT).to(device)

            disp_pred, class_pred, motion_pred = model(padded)
            optimizer.zero_grad()
            total, parts = compute_losses(
                loss_cfg, disp_gt, valid, cat, disp_pred, non_empty, class_pred,
                motion_gt, motion_pred, trans, inst, device,
            )
            if total is None:
                continue
            total.backward()
            optimizer.step()

            if (i + 1) % 500 == 0:
                torch.save({"epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict()},
                           os.path.join(ckpt_dir, "latest.pt"))

            epoch_loss += float(parts[0])
            n_steps += 1
            global_step = (epoch - 1) * len(trainloader) + i
            if i % log_interval == 0:
                rec = {
                    "type": "step", "epoch": epoch, "step": i + 1,
                    "loss": float(parts[0]), "loss_parts": {
                        "disp": parts[0], "cls": parts[1], "motion": parts[2],
                        "bg_tc": parts[3], "sc": parts[4], "fg_tc": parts[5]},
                    "lr": lr_now, "time": round(time.time() - t_epoch, 3),
                }
                write_jsonl(rec)
                if tb is not None:
                    tb.add_scalar("loss/step", parts[0], global_step)

        val_res = None
        if epoch == num_epochs or epoch % val_interval == 0:
            if cfg["data"].get("val_root"):
                val_res = evaluate(
                    model, cfg["data"]["val_root"], split="val", out_dir=None,
                    batch_size=cfg["train"].get("val_batch_size", 4), device=device,
                )

        epoch_rec = {
            "type": "epoch", "epoch": epoch,
            "train_loss": epoch_loss / max(n_steps, 1),
            "val": None if val_res is None else {
                "epe": {k: v["mean"] for k, v in val_res["epe"].items()},
                "OA": val_res["cls"]["OA"], "MCA": val_res["cls"]["MCA"],
            },
            "time": round(time.time() - t_epoch, 3),
        }
        write_jsonl(epoch_rec)
        if tb is not None:
            tb.add_scalar("loss/epoch", epoch_rec["train_loss"], epoch)
            if val_res is not None:
                for k, v in epoch_rec["val"]["epe"].items():
                    if v is not None:
                        tb.add_scalar(f"val_epe/{k}", v, epoch)

        # -- ckpt：epoch 1、每 5 个、>20 的每个 epoch、last.pt
        if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs or epoch > 20:
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": epoch_loss / max(n_steps, 1),
            }
            torch.save(save_dict, os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))
            torch.save(save_dict, os.path.join(ckpt_dir, "last.pt"))

    manifest["end"] = datetime.datetime.now().isoformat()
    manifest["wall_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    _registry_append({
        "run_id": manifest["run_id"], "exp_id": cfg["run"].get("exp_id", ""), "seed": seed,
        "status": "done", "config": cfg.get("_config_path", ""),
        "exp_path": run_dir, "note": f"wall {manifest['wall_seconds']}s",
    })
    if tb is not None:
        tb.close()
    return run_dir


def sys_argv():
    import sys

    return sys.argv[1:]
