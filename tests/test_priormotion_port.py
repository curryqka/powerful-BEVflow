"""PriorMotion 移植对拍：旧 PriorMotionSTPN 与移植版同权重同输入（含 VAE 采样与
teacher-forcing 路径），输出必须逐值一致。"""
import os
import sys

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_PM = os.path.join(CODE, "third_party", "MotionNet", "powerfulmotion")
for p in (CODE, UPSTREAM_PM, os.path.join(UPSTREAM_PM, "nuscenes-devkit", "python-sdk")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _build_batch(device):
    from bevflow.datasets.motionnet import TrainDatasetMultiSeq

    dataset = TrainDatasetMultiSeq(
        dataset_root="/media/nas_users/kaneqian/data/bevflow/preprocessed/mini/train",
        future_frame_skip=0, num_past_frames=5, num_future_frames=20, num_category=5,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    (padded, disp_gt, valid, non_empty, cat, trans, motion_gt, inst, npf, nfu) = next(iter(loader))
    T = int(nfu[0])
    bevs = padded.view(-1, int(npf[0]), 256, 256, 13).to(device)
    # scene_gt = cat(cls 5ch, disp 2T ch, motion 2ch)，与旧代码 train_VAE.py 的 disp_gt_map 一致
    cls_map = cat.permute(0, 1, 4, 2, 3).reshape(-1, 5, 256, 256)
    disp_map = disp_gt.permute(0, 1, 4, 2, 3, 5).reshape(-1, 2 * T, 256, 256)
    motion_map = motion_gt.permute(0, 1, 4, 2, 3).reshape(-1, 2, 256, 256)
    scene_gt = torch.cat([cls_map, disp_map, motion_map], dim=1).to(device)
    return bevs, scene_gt, T


def test_priormotion_port_matches_upstream():
    device = torch.device("cuda")

    sys.argv = ["train_VAE.py", "-d", "x", "--batch", "1", "--nepoch", "1", "--nworker", "0"]
    from model.priormotion import PriorMotionSTPN as UpModel

    from bevflow.models.priormotion import PriorMotionSTPN as MyModel

    torch.manual_seed(0)
    up = UpModel(out_seq_len=20, motion_category_num=2, height_feat_size=13)
    my = MyModel(out_seq_len=20, motion_category_num=2, height_feat_size=13)
    missing, unexpected = my.load_state_dict(up.state_dict(), strict=True)
    assert not missing and not unexpected
    up = up.to(device).train()
    my = my.to(device).train()

    bevs, scene_gt, T = _build_batch(device)

    # teacher-forcing 路径（VAE 采样受种子控制，两边必须一致）
    torch.manual_seed(42)
    out_up = up(bevs, disp_gt_map=scene_gt)
    torch.manual_seed(42)
    out_my = my(bevs, disp_gt_map=scene_gt)

    for i, (a, b) in enumerate(zip(out_up[:3], out_my[:3])):
        assert torch.allclose(a, b, rtol=0, atol=1e-6), f"output {i} mismatch: {(a - b).abs().max().item()}"
    for k in out_up[3]:
        assert torch.allclose(out_up[3][k], out_my[3][k], rtol=0, atol=1e-6), f"vae_dict {k} mismatch"

    # 推理路径（disp_gt_map=None）
    torch.manual_seed(43)
    inf_up = up(bevs, disp_gt_map=None)
    torch.manual_seed(43)
    inf_my = my(bevs, disp_gt_map=None)
    for i, (a, b) in enumerate(zip(inf_up[:3], inf_my[:3])):
        assert torch.allclose(a, b, rtol=0, atol=1e-6), f"inference output {i} mismatch"
