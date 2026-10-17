"""一致性损失（bg_tc / instance sc+fg）开启时的双实现对拍。

上游 compute_and_bp_loss 与 bevflow compute_losses 在同一 batch、同一权重、
LossConfig 全开下，六个损失值必须一致（1e-6）。这是对 R1A 塌缩问题的归因前提：
若一致，则塌缩是配方动力学（零解吸引），而非移植错误。
"""
import os

import numpy as np
import torch
from torch import nn

from bevflow.datasets.motionnet import TrainDatasetMultiSeq

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(CODE, "third_party", "MotionNet")
HEIGHT_FEAT = 13

VOXEL = (0.25, 0.25, 0.4)
EXTENTS = np.array([[-32.0, 32.0], [-32.0, 32.0], [-3.0, 2.0]])


def _make_model_and_batch(device):
    import sys

    sys.argv = ["train_multi_seq.py", "-d", "x", "--batch", "1", "--nepoch", "1", "--nworker", "0"]
    import train_multi_seq as tm
    from model import MotionNet as UpMotionNet

    torch.manual_seed(0)
    up_model = nn.DataParallel(UpMotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT))
    state = up_model.state_dict()

    from bevflow.models.motionnet import MotionNet

    my_model = nn.DataParallel(MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT))
    my_model.load_state_dict(state)
    up_model = up_model.to(device).train()
    my_model = my_model.to(device).train()

    dataset = TrainDatasetMultiSeq(
        dataset_root="/media/nas_users/kaneqian/data/bevflow/preprocessed/mini/train",
        future_frame_skip=0, voxel_size=VOXEL, area_extents=EXTENTS,
        num_past_frames=5, num_future_frames=20, num_category=5,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    return tm, up_model, my_model, batch


def test_consistency_losses_match_upstream():
    device = torch.device("cuda")
    tm, up_model, my_model, batch = _make_model_and_batch(device)
    (padded, disp_gt, valid, non_empty, cat, trans, motion_gt, inst, npf, nfu) = batch
    padded = padded.view(-1, npf[0].item(), 256, 256, HEIGHT_FEAT).to(device)

    optimizer_up = torch.optim.Adam(up_model.parameters(), lr=0.0)
    criterion = nn.SmoothL1Loss(reduction="none")

    disp_u, cell_u, motion_u = up_model(padded)  # 带梯度：上游函数内部会 backward
    tm.use_bg_temporal_consistency = True
    tm.use_fg_temporal_consistency = True
    tm.use_spatial_consistency = True
    tm.use_nn_sampling = False
    values_up = tm.compute_and_bp_loss(
        optimizer_up, device, nfu[0].item(),
        disp_gt, valid, cat, disp_u, criterion, non_empty, cell_u,
        motion_gt, motion_u, trans, inst,
    )
    assert values_up[0] is not None

    # 上游 compute_and_bp_loss 内部做了 backward+step，会改 up_model 权重；
    # 我的实现需要同一步长下的等价梯度路径。为此单独用梯度外置方式再跑一次上游：
    # 直接调用其内部逻辑不可行（闭包全局），改为：重置权重后用我的实现 + 手动 step，
    # 但上游返回的损失值在 backward 前已定，backward/step 不影响返回值，故直接比较。
    from bevflow.models.losses import LossConfig, compute_losses

    with torch.no_grad():
        disp_m, cell_m, motion_m = my_model(padded)
    optimizer_my = torch.optim.Adam(my_model.parameters(), lr=0.0)
    cfg = LossConfig(use_bg_tc=True, use_fg_tc=True, use_sc=True)
    total, values_my = compute_losses(
        cfg, disp_gt, valid, cat, disp_m, non_empty, cell_m,
        motion_gt, motion_m, trans, inst, device,
    )

    names = ["disp", "cls", "motion", "bg_tc", "sc", "fg_tc"]
    for n, got, want in zip(names, values_my, values_up):
        assert got is not None and want is not None, n
        assert abs(got - want) <= 1e-6 * max(1.0, abs(want)), f"{n}: mine={got}, upstream={want}"
