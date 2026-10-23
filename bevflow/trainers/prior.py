"""PriorMotion（DSpG）训练器。相对 multi_seq 的结构性差异只有两处：

1. 训练前向走 teacher-forcing：把 scene_gt（cls 5ch + disp 2T ch + motion 2ch）传进模型
   的 disp_gt_map，未来后验采样替换观测先验采样；评测时 disp_gt_map=None。
2. 总损失 = aux_re_weight * aux + kld_weight * KL(future‖present)。

其余（数据、优化器、调度、ckpt 策略、日志、resume、registry）与 multi_seq 相同。
"""
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

from bevflow.models.losses import LossConfig, compute_losses, kl_future_present
from bevflow.models.priormotion import PriorMotionSTPN
from bevflow.utils.seed import fix_all_seeds

HEIGHT_FEAT = 13


def _git_info(code_dir):
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_dir, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--short"], cwd=code_dir, capture_output=True, text=True).stdout.strip()
    return sha, bool(dirty)


def find_latest_ckpt(ckpt_dir):
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


def build_scene_gt(cat, disp_gt, motion_gt):
    """scene_gt：cls 5ch + disp 2T ch + motion 2ch，展平 adj 维。"""
    b, adj = cat.shape[0], cat.shape[1]
    cls_map = cat.permute(0, 1, 4, 2, 3).reshape(b * adj, 5, 256, 256)
    t = disp_gt.shape[2]
    disp_map = disp_gt.permute(0, 1, 4, 2, 3, 5).reshape(b * adj, 2 * t, 256, 256)
    motion_map = motion_gt.permute(0, 1, 4, 2, 3).reshape(b * adj, 2, 256, 256)
    return torch.cat([cls_map, disp_map, motion_map], dim=1)


def train_prior(cfg, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(os.path.join(logs_dir, "tb"), exist_ok=True)

    seed = cfg["train"].get("seed", 0)
    fix_all_seeds(seed)
    with open(os.path.join(run_dir, "config_snapshot.yaml"), "w") as f:
        json.dump(cfg, f, indent=1, default=str)

    code_dir = cfg["_code_dir"]
    sha, dirty = _git_info(code_dir)
    if dirty:
        diff = subprocess.run(["git", "diff"], cwd=code_dir, capture_output=True, text=True).stdout
        with open(os.path.join(run_dir, "code.diff"), "w") as f:
            f.write(diff)
    manifest = {
        "run_id": cfg["run"]["run_id"], "git": sha, "git_dirty": dirty,
        "config_sha": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest(),
        "data_card_sha": None, "seed": seed, "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "start": datetime.datetime.now().isoformat(), "cmd": " ".join(sys.argv[1:]),
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    from scripts.registry import append as _registry_append

    _registry_append({
        "run_id": manifest["run_id"], "exp_id": cfg["run"].get("exp_id", ""), "seed": seed,
        "status": "running", "config": cfg.get("_config_path", ""), "exp_path": run_dir,
    })

    from bevflow.datasets.motionnet import AREA_EXTENTS, VOXEL_SIZE, TrainDatasetMultiSeq

    card_path = os.path.join(os.path.dirname(cfg["data"]["train_root"]), "data_card.json")
    if os.path.exists(card_path):
        with open(card_path) as f:
            card = json.load(f)
        p = card["params"]
        assert list(p["voxel_size"]) == list(VOXEL_SIZE) and p["height_bins"] == HEIGHT_FEAT
        assert p["past_frames"] == 5 and p["future_frames"] == 20

    trainset = TrainDatasetMultiSeq(
        dataset_root=cfg["data"]["train_root"], future_frame_skip=0,
        voxel_size=VOXEL_SIZE, area_extents=AREA_EXTENTS,
        num_past_frames=5, num_future_frames=20, num_category=5,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"].get("workers", 4))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_rvpe = cfg["model"].get("name") == "priormotion_rvpe"
    model = nn.DataParallel(PriorMotionSTPN(
        out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT,
        fusion=cfg["model"].get("fusion", "repeat"),
        sgru_hidden_init=cfg["model"].get("sgru_hidden_init", "x6"),
        latent_levels=cfg["model"].get("latent_levels", 4),
        use_rvpe=use_rvpe, num_grid=cfg["model"].get("num_grid", 20),
    ))
    model = model.to(device)
    criterion = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=cfg["optim"].get("lr", 0.0016))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg["optim"].get("milestones", [10, 20, 30, 40]), gamma=0.5)

    start_epoch = 1
    if cfg["train"].get("resume", "auto") == "auto":
        latest, latest_n = find_latest_ckpt(ckpt_dir)
        latest_pt = os.path.join(ckpt_dir, "latest.pt")
        if os.path.exists(latest_pt):
            # 实例停止保险：每 500 微批滚动保存。恢复时模型/优化器状态续接，
            # 该 epoch 的数据从头重过一遍（随机顺序下无害）。
            checkpoint = torch.load(latest_pt, map_location="cpu", weights_only=False)
            if latest is None or checkpoint["epoch"] >= latest_n:
                start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print(f"resumed mid-epoch state from {latest_pt} at epoch {checkpoint['epoch']}")
        elif latest is not None:
            checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
            start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"resumed from {latest} at epoch {latest_n}")
            with open(os.path.join(logs_dir, "train.jsonl"), "a") as f:
                f.write(json.dumps({"type": "resume", "step_epoch": latest_n}) + "\n")

    loss_cfg = LossConfig(**{k: v for k, v in cfg.get("loss", {}).items()
                             if k in ("use_bg_tc", "use_fg_tc", "use_sc", "nn_sampling",
                                      "reg_weight_bg_tc", "reg_weight_fg_tc", "reg_weight_sc")})
    aux_re_weight = cfg["loss"].get("aux_re_weight", 5.0)
    kld_weight = cfg["loss"].get("kld_weight", 0.5)
    accum_steps = cfg["train"].get("accum_steps", 1)

    train_jsonl = os.path.join(logs_dir, "train.jsonl")
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(os.path.join(logs_dir, "tb"))
    except ImportError:
        tb = None
    log_interval = cfg["log"].get("interval", 50)
    val_interval = cfg["train"].get("val_interval", 5)
    num_epochs = cfg["train"]["epochs"]

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
            (padded, disp_gt, valid, non_empty, cat, trans, motion_gt, inst, npf, nfu) = data
            padded = padded.view(-1, npf[0].item(), 256, 256, HEIGHT_FEAT).to(device)
            scene_gt = build_scene_gt(cat, disp_gt, motion_gt).to(device)

            instance_info = None
            if use_rvpe:
                from bevflow.preprocess.instances import get_instance_info_paper

                b_adj = disp_gt.shape[0] * disp_gt.shape[1]
                t_frames = disp_gt.shape[2]
                instance_info = get_instance_info_paper(
                    disp_gt.permute(0, 1, 2, 5, 3, 4).reshape(b_adj, t_frames, 2, 256, 256).numpy(),
                    cat.permute(0, 1, 4, 2, 3).reshape(b_adj, 5, 256, 256).numpy(),
                    inst.reshape(b_adj, 256, 256).numpy(),
                    num_grid=cfg["model"].get("num_grid", 20),
                )

            disp_pred, class_pred, motion_pred, vae_dict = model(
                padded, disp_gt_map=scene_gt, instance_info=instance_info)
            if i % accum_steps == 0:
                optimizer.zero_grad()
            aux, parts = compute_losses(
                loss_cfg, disp_gt, valid, cat, disp_pred, non_empty, class_pred,
                motion_gt, motion_pred, trans, inst, device)
            if aux is None:
                continue
            kl = kl_future_present(vae_dict["present_mu"], vae_dict["present_log_sigma"],
                                   vae_dict["future_mu"], vae_dict["future_log_sigma"])
            total = (aux_re_weight * aux + kld_weight * kl) / accum_steps
            total.backward()
            if (i + 1) % accum_steps == 0 or i + 1 == len(trainloader):
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
                rec = {"type": "step", "epoch": epoch, "step": i + 1,
                       "loss": float(total), "loss_parts": {
                           "disp": parts[0], "cls": parts[1], "motion": parts[2],
                           "bg_tc": parts[3], "sc": parts[4], "fg_tc": parts[5],
                           "kl": float(kl)},
                       "lr": lr_now, "time": round(time.time() - t_epoch, 3)}
                write_jsonl(rec)
                if tb is not None:
                    tb.add_scalar("loss/step", float(total), global_step)

        val_res = None
        if epoch == num_epochs or epoch % val_interval == 0:
            if cfg["data"].get("val_root"):
                val_res = evaluate(model, cfg["data"]["val_root"], split="val", out_dir=None,
                                   batch_size=cfg["train"].get("val_batch_size", 4), device=device)

        epoch_rec = {"type": "epoch", "epoch": epoch,
                     "train_loss": epoch_loss / max(n_steps, 1),
                     "val": None if val_res is None else {
                         "epe": {k: v["mean"] for k, v in val_res["epe"].items()},
                         "OA": val_res["cls"]["OA"], "MCA": val_res["cls"]["MCA"]},
                     "time": round(time.time() - t_epoch, 3)}
        write_jsonl(epoch_rec)
        if tb is not None:
            tb.add_scalar("loss/epoch", epoch_rec["train_loss"], epoch)

        if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs or epoch > 20:
            save_dict = {"epoch": epoch,
                         "model_state_dict": model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(),
                         "scheduler_state_dict": scheduler.state_dict(),
                         "loss": epoch_loss / max(n_steps, 1)}
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
