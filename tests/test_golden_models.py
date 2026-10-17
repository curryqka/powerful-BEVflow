"""golden 对拍：前向（官方权重）与损失值，与重构前捕获的基线比较。"""
import json
import os

import numpy as np
import torch
from torch import nn

from bevflow.models.motionnet import MotionNet

GOLDEN = "/media/nas_users/kaneqian/data/bevflow/golden"
OFFICIAL_CKPT = "/media/nas_users/kaneqian/Experiment/bevflow/_models/motionnet_official/model.pth"
HEIGHT_FEAT = 13


def test_forward_official_weights():
    d = np.load(os.path.join(GOLDEN, "forward.npz"))
    model = nn.DataParallel(MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT))
    ckpt = torch.load(OFFICIAL_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.cuda().eval()

    with torch.no_grad():
        disp, cell, motion = model(torch.from_numpy(d["x"]).cuda())

    assert np.allclose(disp.cpu().numpy(), d["disp"], rtol=0, atol=1e-5), "disp head mismatch"
    assert np.allclose(cell.cpu().numpy(), d["cell"], rtol=0, atol=1e-5), "cell head mismatch"
    assert np.allclose(motion.cpu().numpy(), d["motion"], rtol=0, atol=1e-5), "motion head mismatch"


def test_loss_values():
    from bevflow.models.losses import LossConfig, compute_losses

    z = np.load(os.path.join(GOLDEN, "loss_inputs.npz"))
    golden = json.load(open(os.path.join(GOLDEN, "loss.json")))["losses"]

    device = torch.device("cuda")
    model = nn.DataParallel(MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT)).to(device)
    weights = z["init_weights"]
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data = torch.from_numpy(np.frombuffer(weights, dtype=np.float32, count=n, offset=offset).copy()
                                  ).view_as(p).to(device)
        offset += n * 4

    padded = torch.from_numpy(z["padded"])
    npf = torch.from_numpy(z["npf"])
    with torch.no_grad():
        disp_pred, cell_pred, motion_pred = model(
            padded.view(-1, int(npf[0]), 256, 256, HEIGHT_FEAT).to(device)
        )

    cfg = LossConfig()  # 上游默认：一致性损失全关
    total, parts = compute_losses(
        cfg,
        torch.from_numpy(z["disp_gt"]), torch.from_numpy(z["valid"]),
        torch.from_numpy(z["cat"]), disp_pred, torch.from_numpy(z["non_empty"]),
        cell_pred, torch.from_numpy(z["motion_gt"]), motion_pred,
        torch.from_numpy(z["trans"]), torch.from_numpy(z["inst"]), device,
    )
    assert parts[0] is not None
    for got, want in zip(parts, golden):
        assert abs(got - want) < 1e-6, f"loss mismatch: got {got}, want {want}"
