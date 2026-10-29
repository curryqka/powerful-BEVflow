# SPDX-License-Identifier: Apache-2.0
"""RVpE 积木：从旧代码 model/prior_extractor.py 搬运（SimplePositionEncoding /
SineContinuousPositionalEncoding / DispEncoding / GridInsTransformer(ISA) /
MaskedCrossAttention(PCA)）。行为保持，仅清理注释与死代码。
"""
import numpy as np
import torch
import torch.nn as nn


class SimplePositionEncoding(nn.Module):
    """BEV 侧可学习行列位置编码，返回 [bs, num_feats*2, h, w]。"""

    def __init__(self, num_feats, row_num_embed=50, col_num_embed=50):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats

    def forward(self, mask):
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat((x_embed.unsqueeze(0).repeat(h, 1, 1),
                         y_embed.unsqueeze(1).repeat(1, w, 1)), dim=-1)
        return pos.permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)


class SineContinuousPositionalEncoding(nn.Module):
    """连续坐标的正弦编码：[B, N, D] → [B, N, D*num_feats]。"""

    def __init__(self, num_feats, temperature=10000, normalize=False, range=None,
                 scale=2 * np.pi, offset=0.):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.range = torch.tensor(range) if range is not None else None
        self.offset = torch.tensor(offset) if offset is not None else None
        self.scale = scale

    def forward(self, x):
        B, N, D = x.shape
        if self.normalize:
            x = (x - self.offset.to(x.device)) / self.range.to(x.device) * self.scale
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x[..., None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=3)
        return pos_x.view(B, N, D * self.num_feats)


class DispEncoding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.embedding_layer = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.embedding_layer(x)


class GridInsTransformer(nn.Module):
    """VpE 的 ISA：实例序列 + 位移 + 类别 → 每实例 dmodel 维特征。批次为 list（ragged）。"""

    def __init__(self, input_dim=30, dmodel=256, hidden_dim=2048, nheads=8, nlayers=6,
                 batch_first=False, pos_encoder=None, disp_encoder=None):
        super().__init__()
        self.batch_dim = 0 if batch_first else 1
        self.map_embedding = nn.Linear(input_dim, dmodel)
        self.use_positional_encoding = pos_encoder is not None
        if self.use_positional_encoding:
            self.pos_encoder = SineContinuousPositionalEncoding(
                num_feats=16, temperature=1000, normalize=False, range=[256, 256], offset=[0, 0])
        self.use_disp_encoding = disp_encoder is not None
        if self.use_disp_encoding:
            self.disp_encoder = DispEncoding(input_dim=2, embedding_dim=16 * 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dmodel, nhead=nheads, dim_feedforward=hidden_dim, dropout=0.1,
            layer_norm_eps=1e-5, batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(self, grid_ins, disp_ins, onehot_category, device=None):
        batch_graph_feats = []
        for graph_polylines, disp, onehot_cat in zip(grid_ins, disp_ins, onehot_category):
            graph_polylines = torch.as_tensor(graph_polylines, dtype=torch.float32, device=device)
            disp = torch.as_tensor(disp, dtype=torch.float32, device=device)
            onehot_cat = torch.as_tensor(onehot_cat, dtype=torch.float32, device=device)
            if self.use_positional_encoding:
                graph_polylines = self.pos_encoder(graph_polylines)
            if self.use_disp_encoding:
                disp = self.disp_encoder(disp)
            npolylines, npoints, pdim = graph_polylines.shape
            nins, ngrids, ddim = disp.shape
            if onehot_cat.shape[1] == 0:
                graph_feat = graph_polylines.view(npolylines, npoints * pdim)
            elif self.use_disp_encoding:
                graph_feat = torch.cat([graph_polylines.view(npolylines, npoints * pdim),
                                        disp.view(nins, ngrids * ddim), onehot_cat], dim=-1)
            else:
                graph_feat = torch.cat([graph_polylines.view(npolylines, npoints * pdim),
                                        onehot_cat], dim=-1)
            graph_feat = self.map_embedding(graph_feat)
            graph_feat = self.transformer_encoder(graph_feat.unsqueeze(self.batch_dim))
            batch_graph_feats.append(graph_feat.squeeze(self.batch_dim))
        return batch_graph_feats


class MaskedCrossAttention(nn.Module):
    """PCA：BEV 特征（query）对实例特征（key/value）的带 padding mask 交叉注意力。"""

    def __init__(self, embed_dims=256, num_heads=8, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.multihead_attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=batch_first)
        self.norm = nn.LayerNorm(embed_dims, eps=1e-5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, residual=None, query_pos=None,
                spatial_shapes=None, use_scale_factor=False):
        if key is None:
            key = value
        if residual is None:
            residual = query
        if query_pos is not None:
            query = query + query_pos

        bs, _, _ = query.shape
        key_padding_mask = (
            torch.arange(spatial_shapes.max(), device=spatial_shapes.device).unsqueeze(0).repeat(bs, 1)
            >= spatial_shapes.unsqueeze(1)
        ).to(torch.bool)
        if use_scale_factor:
            query = query * torch.tensor(self.embed_dims ** -0.5, device=query.device)

        attn_output, _ = self.multihead_attn(
            query=query, key=key, value=value, key_padding_mask=key_padding_mask, need_weights=True)
        return residual + self.dropout(attn_output)


class VpE(nn.Module):
    """论文式 5 的矢量先验编码：沿轨迹采样的 N 点（正弦位置编码）+ one-hot 类别 +
    逐帧位移序列的 LSTM 输出，拼接后 MLP → 实例 token，再过 L=6 的 Transformer 编码器（ISA）。"""

    def __init__(self, out_seq_len=20, num_grid=20, cat_num=5, d_m=16,
                 dmodel=256, hidden_dim=2048, nheads=8, nlayers=6):
        super().__init__()
        self.out_seq_len = out_seq_len
        self.num_grid = num_grid
        self.pos_encoder = SineContinuousPositionalEncoding(
            num_feats=16, temperature=1000, normalize=False, range=[256, 256], offset=[0, 0])
        self.lstm = nn.LSTM(input_size=2, hidden_size=d_m, num_layers=1, batch_first=True)
        self.map_embedding = nn.Linear(num_grid * 32 + cat_num + out_seq_len * d_m, dmodel)
        layer = nn.TransformerEncoderLayer(d_model=dmodel, nhead=nheads,
                                           dim_feedforward=hidden_dim, dropout=0.1,
                                           layer_norm_eps=1e-5, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=nlayers)

    def forward(self, traj_list, disp_seq_list, onehot_list, device=None):
        feats = []
        for traj, dseq, oh in zip(traj_list, disp_seq_list, onehot_list):
            n = traj.shape[0]
            pos = self.pos_encoder(torch.as_tensor(traj, dtype=torch.float32, device=device))
            h, _ = self.lstm(torch.as_tensor(dseq, dtype=torch.float32, device=device))
            feat = torch.cat([pos.reshape(n, -1),
                              torch.as_tensor(oh, dtype=torch.float32, device=device),
                              h.reshape(n, -1)], dim=-1)
            feat = self.transformer_encoder(self.map_embedding(feat).unsqueeze(0))
            feats.append(feat.squeeze(0))
        return feats
