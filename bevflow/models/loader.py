"""统一 ckpt 加载。官方权重与训练 ckpt 都是 DataParallel 键名（module. 前缀）。"""
import torch
from torch import nn

from bevflow.models.motionnet import MotionNet

HEIGHT_FEAT = 13


def load_model(ckpt_path, device, name="motionnet"):
    if name == "priormotion":
        from bevflow.models.priormotion import PriorMotionSTPN

        model = nn.DataParallel(PriorMotionSTPN(out_seq_len=20, motion_category_num=2,
                                                height_feat_size=HEIGHT_FEAT))
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
