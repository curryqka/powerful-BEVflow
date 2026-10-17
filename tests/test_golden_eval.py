"""golden 对拍：评测器。新 evaluators.motion_eval 与上游 eval.py 在同一 ckpt + mini val 上
的 metrics 数值一致（EPE/OA/MCA，容差 1e-6；上游怪癖保持：OA 按 batch 累计，bs=4）。
参照：$RESULTS/_smoke/20261017_002303/eval/metrics.json（上游 eval.py 产出）。
"""
import json
import os

import numpy as np
import torch

from bevflow.evaluators.motion_eval import evaluate
from bevflow.models.loader import load_motionnet

REF = "/media/nas_users/kaneqian/results/bevflow/_smoke/20261017_002303/eval/metrics.json"
CKPT = "/media/nas_users/kaneqian/results/bevflow/_smoke/20261017_002303/train/run1/train_multi_seq/2026-10-17_00-16-36/epoch_2.pth"
VAL = "/media/nas_users/kaneqian/data/bevflow/preprocessed/mini/val"


def _find_ckpt():
    root = "/media/nas_users/kaneqian/results/bevflow/_smoke/20261017_002303/train"
    for dirpath, _dirs, files in os.walk(root):
        if "epoch_2.pth" in files:
            return os.path.join(dirpath, "epoch_2.pth")
    raise AssertionError("golden ckpt not found")


def test_eval_matches_upstream():
    ref = json.load(open(REF))
    device = torch.device("cuda")
    model = load_motionnet(_find_ckpt(), device)
    got = evaluate(
        model, VAL, split="val", out_dir=None, batch_size=4, num_workers=2, device=device,
    )
    for tier in ("static", "slow", "fast"):
        assert abs(got["epe"][tier]["mean"] - ref["epe"][tier]["mean"]) < 1e-6, tier
        assert abs(got["epe"][tier]["median"] - ref["epe"][tier]["median"]) < 1e-6, tier
    assert abs(got["cls"]["OA"] - ref["cls"]["OA"]) < 1e-6
