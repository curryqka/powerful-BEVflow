"""LEGO-Motion（IROS 2025）实现。规格见 docs/design/spec_legomotion.md。

结构：BEV Encoder（STPN 复用 motionnet.py）→ IaIE（式1-2）→ IeBE（早期式3-4 + 晚期式5-6）
→ 任务头（三头沿用 MotionNet 惯例 + 掩码头）。关键默认值来自用户记忆，均可配置：
N=256 实例 query，D=64，BEV 降采样 16×16 作 IaIE 的 key/value。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bevflow.models.motionnet import CellClassification, MotionPrediction, StateEstimation, STPN

HEIGHT_FEAT = 13
CAT_NUM = 5


class PreLNBlock(nn.Module):
    """Pre-LN Transformer block（自注意力或交叉注意力 + FFN + 残差）。"""

    def __init__(self, dim, nheads, ffn_dim, is_cross=False, dropout=0.0):
        super().__init__()
        self.is_cross = is_cross
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nheads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(True), nn.Linear(ffn_dim, dim))
        self.norm_q = nn.LayerNorm(dim)

    def forward(self, x, kv=None):
        if self.is_cross:
            q = self.norm_q(x)
            k = self.norm1(kv)
            x = x + self.attn(q, k, k, need_weights=False)[0]
        else:
            y = self.norm1(x)
            x = x + self.attn(y, y, y, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class IaIE(nn.Module):
    """实例交互编码器（式1-2）：N 个可学习 query，自注意力 + 与 16×16 BEV 的交叉注意力。"""

    def __init__(self, dim=64, n_queries=256, scale=16, nheads=4, ffn_dim=256, nlayers=1):
        super().__init__()
        self.query_embed = nn.Embedding(n_queries, dim)
        self.bev_down = nn.Sequential(
            nn.Conv2d(32, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((scale, scale)),
        )
        self.blocks = nn.ModuleList([
            PreLNBlock(dim, nheads, ffn_dim, is_cross=False),
            PreLNBlock(dim, nheads, ffn_dim, is_cross=True),
        ])
        self.dim = dim

    def forward(self, bev):
        b = bev.size(0)
        q = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        b_small = self.bev_down(bev).flatten(2).permute(0, 2, 1)  # (B, 256, D)
        q = self.blocks[0](q)
        q = self.blocks[1](q, kv=b_small)
        return q  # (B, N, D)


class IeBE(nn.Module):
    """实例增强 BEV 编码器：SE 早期融合（式3-4）+ 线性投影与交叉注意力晚期融合（式5-6）。"""

    def __init__(self, dim=64, nheads=4, ffn_dim=256):
        super().__init__()
        self.proj_bev = nn.Conv2d(32, dim, kernel_size=1)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1), nn.Sigmoid())
        self.ins_norm = nn.LayerNorm(dim)
        self.ins_self = nn.MultiheadAttention(dim, nheads, batch_first=True)
        self.phi_bev = nn.Linear(dim, dim)
        self.phi_ins = nn.Linear(dim, dim)
        self.late = PreLNBlock(dim, nheads, ffn_dim, is_cross=True)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(True), nn.Linear(ffn_dim, dim))
        self.dim = dim

    def forward(self, bev, ins_feat):
        b, _, h, w = bev.shape
        bd = self.proj_bev(bev)                       # (B, D, H, W)
        f = bd * self.se(bd)                          # SE（式3）
        flat = f.flatten(2).permute(0, 2, 1)          # (B, HW, D)

        ins = self.ins_norm(ins_feat)
        ins = ins + self.ins_self(ins, ins, ins, need_weights=False)[0]
        logits = torch.bmm(flat, ins.transpose(1, 2))  # (B, HW, N)
        gate = torch.sigmoid(logits.max(dim=-1).values)  # max over N（式4）
        f_early = flat * gate.unsqueeze(-1)

        f_star = self.phi_bev(f_early)
        ins_star = self.phi_ins(ins)
        e = self.late(f_star, kv=ins_star)            # 式6 晚期交叉注意力
        e = e + self.ffn(e)
        return e.permute(0, 2, 1).view(b, self.dim, h, w)  # (B, D, H, W)


class MaskHead(nn.Module):
    """掩码头：mask logits = I_i · B（sigmoid 前），类别 MLP 输出 Kc+1。"""

    def __init__(self, dim=64, num_classes=CAT_NUM + 1, ffn_dim=256):
        super().__init__()
        self.cls_head = nn.Sequential(nn.Linear(dim, ffn_dim), nn.ReLU(True), nn.Linear(ffn_dim, num_classes))

    def forward(self, ins_feat, bev_proj):
        # ins_feat (B,N,D), bev_proj (B,D,H,W) → masks (B,N,HW), classes (B,N,Kc+1)
        b, n, d = ins_feat.shape
        masks = torch.einsum("bnd,bdhw->bnhw", ins_feat, bev_proj).flatten(2)
        return masks, self.cls_head(ins_feat)


class LegoMotion(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT,
                 dim=64, n_queries=256, bev_scale=16, nheads=4, use_checkpoint=False):
        super().__init__()
        self.out_seq_len = out_seq_len
        self.stpn = STPN(height_feat_size=height_feat_size)
        self.iaie = IaIE(dim=dim, n_queries=n_queries, scale=bev_scale, nheads=nheads)
        self.iebe = IeBE(dim=dim, nheads=nheads)
        self.to_grid = nn.Conv2d(dim, 32, kernel_size=1)
        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)
        self.mask_head = MaskHead(dim=dim)

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)
        x = self.stpn(bevs)                       # (B, 32, 256, 256)
        ins = self.iaie(x)                        # (B, N, D)
        e = self.iebe(x, ins)                     # (B, D, 256, 256)
        g = self.to_grid(e)                       # (B, 32, 256, 256)

        disp = self.motion_pred(g)
        disp = disp.reshape(-1, 2, 256, 256)
        masks, classes = self.mask_head(ins, self.iebe.proj_bev(x))
        return disp, self.cell_classify(g), self.state_classify(g), masks, classes
