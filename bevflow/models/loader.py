"""统一 ckpt 加载。官方权重与训练 ckpt 都是 DataParallel 键名（module. 前缀）。"""
import torch
from torch import nn

from bevflow.models.motionnet import MotionNet

HEIGHT_FEAT = 13


def load_model(ckpt_path, device, name="motionnet"):
    if name == "priormotion":
        import os

        import yaml

        from bevflow.models.priormotion import PriorMotionSTPN

        # 架构探索开关从 run 目录的 config_snapshot 读取，保证变体 ckpt 能正确重建模型
        snap = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path))),
                            "config_snapshot.yaml")
        mc = {}
        if os.path.exists(snap):
            with open(snap) as f:
                mc = (yaml.safe_load(f) or {}).get("model", {}) or {}
        model = nn.DataParallel(PriorMotionSTPN(
            out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT,
            fusion=mc.get("fusion", "repeat"),
            sgru_hidden_init=mc.get("sgru_hidden_init", "x6"),
            latent_levels=mc.get("latent_levels", 4),
        ))
    else:
        model = nn.DataParallel(MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model.to(device).eval()


def load_motionnet(ckpt_path, device):
    return load_model(ckpt_path, device, "motionnet")
