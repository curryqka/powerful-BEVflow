"""CPU 单元测试：所有模型用随机张量过一遍 forward（不依赖 nuScenes/GPU）。"""
import sys

import torch

sys.path.insert(0, ".")

from bevflow.models.legomotion.model import LegoMotion
from bevflow.models.motionnet import MotionNet
from bevflow.models.priormotion import PriorMotionSTPN


def _input():
    return torch.randn(1, 5, 256, 256, 13)


def test_motionnet_cpu():
    m = MotionNet().eval()
    with torch.no_grad():
        disp, cell, state = m(_input())
    assert tuple(disp.shape) == (20, 2, 256, 256)
    assert tuple(cell.shape) == (1, 5, 256, 256)
    assert torch.isfinite(disp).all()


def test_priormotion_cpu():
    m = PriorMotionSTPN().eval()
    with torch.no_grad():
        disp, cell, state, vae = m(_input())
    assert tuple(disp.shape) == (20, 2, 256, 256)
    assert vae is None  # 推理路径不返回 vae_dict


def test_legomotion_cpu():
    m = LegoMotion().eval()
    with torch.no_grad():
        disp, cell, state, masks, classes = m(_input())
    assert tuple(disp.shape) == (20, 2, 256, 256)
    assert tuple(masks.shape) == (1, 256, 65536)
    assert tuple(classes.shape) == (1, 256, 6)
