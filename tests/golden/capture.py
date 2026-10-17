"""捕获 golden 基线：重构前的上游行为快照。

产物放 $DATA/golden/（NAS，不入库），SHA256 记录进 tests/golden/manifest.json（入库）。
三份基线：
1. forward：官方权重 + 固定种子输入，三个头输出。
2. loss：固定合成 pred/GT 过上游 compute_and_bp_loss，六个损失值。
3. preprocess：直接引用冒烟产物 $DATA/preprocessed/mini/train/116_*（上游 gen_data 生成）。
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPSTREAM = os.path.join(CODE, "third_party", "MotionNet")
sys.path.insert(0, UPSTREAM)
sys.path.insert(0, os.path.join(UPSTREAM, "nuscenes-devkit", "python-sdk"))

sys.argv = ["train_multi_seq.py", "-d", "x", "--batch", "1", "--nepoch", "1", "--nworker", "0"]

GOLDEN = "/media/nas_users/kaneqian/data/bevflow/golden"
CKPT = "/media/nas_users/kaneqian/Experiment/bevflow/_models/motionnet_official/model.pth"
HEIGHT_FEAT = 13


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_forward():
    from model import MotionNet

    device = torch.device("cuda")
    model = torch.nn.DataParallel(MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT))
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 5, 256, 256, HEIGHT_FEAT, generator=g)
    with torch.no_grad():
        disp, cell, motion = model(x.to(device))
    out = os.path.join(GOLDEN, "forward.npz")
    np.savez_compressed(
        out,
        x=x.numpy(),
        disp=disp.cpu().numpy(),
        cell=cell.cpu().numpy(),
        motion=motion.cpu().numpy(),
    )
    return out


def capture_loss():
    import train_multi_seq as tm
    from data.nuscenes_dataloader import TrainDatasetMultiSeq

    dataset = TrainDatasetMultiSeq(
        dataset_root="/media/nas_users/kaneqian/data/bevflow/preprocessed/mini/train",
        future_frame_skip=0,
        voxel_size=(0.25, 0.25, 0.4),
        area_extents=np.array([[-32.0, 32.0], [-32.0, 32.0], [-3.0, 2.0]]),
        num_past_frames=5,
        num_future_frames=20,
        num_category=5,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    (padded, disp_gt, valid, non_empty, cat, trans, motion_gt, inst, npf, nfu) = next(iter(loader))

    from model import MotionNet

    device = torch.device("cuda")
    model = torch.nn.DataParallel(
        MotionNet(out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT)
    ).to(device).train()
    g = torch.Generator().manual_seed(1)
    for p_ in model.parameters():
        p_.data = (torch.randn(p_.shape, generator=g) * 0.02).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.SmoothL1Loss(reduction="none")

    padded = padded
    disp_pred, cell_pred, motion_pred = model(padded.view(-1, npf[0].item(), 256, 256, HEIGHT_FEAT).to(device))

    init_weights = b"".join(p_.detach().cpu().numpy().tobytes() for p_ in model.parameters())

    values = tm.compute_and_bp_loss(
        optimizer, device, nfu[0].item(),
        disp_gt, valid, cat, disp_pred, criterion, non_empty, cell_pred,
        motion_gt, motion_pred, trans, inst,
    )
    assert values[0] is not None, "no valid pixels in golden batch"
    out = os.path.join(GOLDEN, "loss.json")
    with open(out, "w") as f:
        json.dump({"losses": [float(v) for v in values]}, f, indent=1)
    np.savez_compressed(
        os.path.join(GOLDEN, "loss_inputs.npz"),
        padded=padded.numpy(), disp_gt=disp_gt.numpy(), valid=valid.numpy(),
        non_empty=non_empty.numpy(), cat=cat.numpy(), trans=trans.numpy(),
        motion_gt=motion_gt.numpy(), inst=inst.numpy(),
        npf=npf.numpy(), nfu=nfu.numpy(),
        init_weights=init_weights,
    )
    return out


if __name__ == "__main__":
    os.makedirs(GOLDEN, exist_ok=True)
    paths = [capture_forward(), capture_loss()]
    manifest = {os.path.basename(p): {"sha256": sha256(p), "size": os.path.getsize(p)} for p in paths}
    manifest["preprocess_ref"] = "/media/nas_users/kaneqian/data/bevflow/preprocessed/mini/train/116_0"
    manifest["eval_ref"] = "/media/nas_users/kaneqian/results/bevflow/_smoke/20261017_002303/eval/metrics.json"
    with open(os.path.join(CODE, "tests", "golden", "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))
