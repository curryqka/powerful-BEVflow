import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from model.attention import CAlayer

class SimplePositionEncoding(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(SimplePositionEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos
class SineContinuousPositionalEncoding(nn.Module):
    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 range=None,
                 scale=2 * np.pi,
                 offset=0.,
                 init_cfg=None):
        super(SineContinuousPositionalEncoding, self).__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.range = torch.tensor(range) if range is not None else None
        self.offset = torch.tensor(offset) if offset is not None else None
        self.scale = scale

    def forward(self, x):
        """
        x: [B, N, D]

        return: [B, N, D * num_feats]
        """
        B, N, D = x.shape
        if self.normalize:
            x = (x - self.offset.to(x.device)) / self.range.to(x.device) * self.scale
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x[..., None] / dim_t  # [B, N, D, num_feats]
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=3).view(B, N, D * self.num_feats)
        return pos_x

class DispEncoding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(DispEncoding, self).__init__()
        self.embedding_layer = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        # 假设输入 x 的形状为 (bs, num_instance * num_grid, 2)
        embedded_x = self.embedding_layer(x)
        return embedded_x

class TemporalDispEncoding(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=32):
        super(TemporalDispEncoding, self).__init__()

        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

        # 定义 MLP 层
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, disp_info):
        # 假设输入 disp_info 的形状为 (bs, seq_len, input_dim)，即 (20, 2)
        bs, seq_len, input_dim = disp_info.shape

        # 通过 LSTM 层
        lstm_out, _ = self.lstm(disp_info)

        # 通过 MLP 层
        mlp_out = self.mlp(lstm_out)

        return mlp_out


class GridInsTransformer(nn.Module):
    def __init__(self,
                 input_dim=30,  # 2 * 11 points + 8 classes
                 dmodel=256,
                 hidden_dim=2048,  # set this to something smaller
                 nheads=8,
                 nlayers=6,
                 batch_first=False,  # set to True
                 pos_encoder=None,
                 disp_encoder=None,
                 **kwargs):
        super(GridInsTransformer, self).__init__(**kwargs)
        self.batch_dim = 0 if batch_first else 1
        self.map_embedding = nn.Linear(input_dim, dmodel)

        if pos_encoder is not None:
            self.use_positional_encoding = True
            # self.pos_encoder = build_positional_encoding(pos_encoder)
            self.pos_encoder = SineContinuousPositionalEncoding(num_feats=16,  # 2 * 16 = 32 final dim
                                                                temperature=1000,
                                                                normalize=False,
                                                                range=[256, 256],
                                                                offset=[0, 0],
                                                                    )
        else:
            self.use_positional_encoding = False

        if disp_encoder=='mlp':
            self.use_disp_encoding = True
            self.disp_encoder = DispEncoding(input_dim=2,
                                             embedding_dim=16*2)
        elif disp_encoder=='temporal':
            self.use_disp_encoding = True
            self.disp_encoder = TemporalDispEncoding(input_dim=2,
                                                     hidden_dim=64,
                                                     output_dim=16*2)
        else:
            self.use_disp_encoding = False

        encoder_layer = nn.TransformerEncoderLayer(d_model=dmodel,
                                                        nhead=nheads,
                                                        dim_feedforward=hidden_dim,
                                                        dropout=0.1,
                                                        layer_norm_eps=1e-5,
                                                        batch_first=batch_first,)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(self, grid_ins, disp_ins, onehot_category, device=None):
        # batch_map_graph: list of polylines
        # batch, num_polylines * points * 2 (x, y)
        # onehot_category: batch, num_polylines * num_categories, onehot encoding of categories
        # TODO: make batched
        batch_graph_feats = []
        for graph_polylines, disp, onehot_cat in zip(grid_ins, disp_ins, onehot_category):
            # if graph_polylines.dtype == np.ndarray:
            graph_polylines = torch.from_numpy(graph_polylines).to(device, dtype=torch.float32)
            disp = torch.from_numpy(disp).to(device, dtype=torch.float32)
            onehot_cat = torch.from_numpy(onehot_cat).to(device, dtype=torch.float32)
            # else:
                # graph_polylines = graph_polylines.to(device, dtype=torch.float32)
                # disp = disp.to(device, dtype=torch.float32)
                # onehot_cat = onehot_cat.to(device, dtype=torch.float32)
            # breakpoint()
            if self.use_positional_encoding:
                # N, _ = graph_polylines.shape
                # graph_polylines = graph_polylines.view(-1, 20, 2)
                graph_polylines = self.pos_encoder(graph_polylines)
            # breakpoint()
            if self.use_disp_encoding:
                disp = self.disp_encoder(disp)
            # breakpoint()
            npolylines, npoints, pdim = graph_polylines.shape
            nins, ngrids, ddim = disp.shape
            # onehot_cat = onehot_cat.view(npolylines, -1)
            if onehot_cat.shape[1] == 0:
                graph_feat = graph_polylines.view(npolylines, npoints * pdim)
            else:
                # breakpoint()
                if self.use_disp_encoding:
                    graph_feat = torch.cat([graph_polylines.view(npolylines, npoints * pdim), \
                        disp.view(nins, ngrids * ddim), onehot_cat], dim=-1)

                else:
                    graph_feat = torch.cat([graph_polylines.view(npolylines, npoints * pdim), \
                        onehot_cat], dim=-1)
            # breakpoint()
            # embed features
            graph_feat = self.map_embedding(graph_feat)  # num_polylines, dmodel

            # transformer encoder
            graph_feat = self.transformer_encoder(graph_feat.unsqueeze(self.batch_dim))  # 1, num_polylines, hidden_dim
            graph_feat = graph_feat.squeeze(self.batch_dim)  # num_polylines, hidden_dim
            batch_graph_feats.append(graph_feat.squeeze(self.batch_dim))
        return batch_graph_feats

class MaskedCrossAttentionV0(nn.Module):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): The number of heads in multihead attention
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.0,
                 init_cfg=None,
                 batch_first=True,
                 **kwargs
                 ):
        super(MaskedCrossAttentionV0, self).__init__()

        self.init_cfg = init_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.multihead_attn = nn.MultiheadAttention(embed_dims,
                                                    num_heads,
                                                    dropout=dropout,
                                                    batch_first=batch_first,
                                                     **kwargs)

    # @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                spatial_shapes=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (bs, max_num_vectors, embed_dims)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            spatial_shapes (Tensor): number of polylines in each batch.
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if key is None:
            key = value

        if residual is None:
            residual = query

        if query_pos is not None:
            query = query + query_pos

        bs, _, _ = query.shape
        key_padding_mask = torch.arange(spatial_shapes.max(), device=spatial_shapes.device).unsqueeze(0).repeat(bs, 1) >= spatial_shapes.unsqueeze(1)

        queries = self.multihead_attn(query=query, key=key, value=value,
                                      key_padding_mask=key_padding_mask, need_weights=False)[0]

        return queries + residual


class MaskedCrossAttention(nn.Module):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): The number of heads in multihead attention
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.0,
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):
        super(MaskedCrossAttention, self).__init__()

        self.init_cfg = init_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        # 多头注意力层
        self.multihead_attn = nn.MultiheadAttention(embed_dims,
                                                    num_heads,
                                                    dropout=dropout,
                                                    batch_first=batch_first,
                                                    **kwargs)

        # 添加 LayerNorm 来稳定训练
        self.norm = nn.LayerNorm(embed_dims, eps=1e-5)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    # @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                spatial_shapes=None,
                use_scale_factor=False,
                debug_mode=False,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (bs, max_num_vectors, embed_dims)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            spatial_shapes (Tensor): number of polylines in each batch.
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if key is None:
            key = value

        if residual is None:
            residual = query

        # 加上位置编码
        if query_pos is not None:
            query = query + query_pos

        # 获取 batch size
        bs, _, _ = query.shape

        # 创建 key padding mask
        key_padding_mask = torch.arange(spatial_shapes.max(), device=spatial_shapes.device).unsqueeze(0).repeat(bs, 1) >= spatial_shapes.unsqueeze(1)

        # 检查输入数据中是否有 nan 值
        assert not torch.isnan(query).any(), "query contains nan"
        assert not torch.isnan(key).any(), "key contains nan"
        assert not torch.isnan(value).any(), "value contains nan"
        if key_padding_mask is not None:
            assert not torch.isnan(key_padding_mask).any(), "key_padding_mask contains nan"
        key_padding_mask = key_padding_mask.to(torch.bool)  # 确保是布尔类型
        # 计算多头注意力
        # attn_output = self.multihead_attn(query=query, key=key, value=value,
        #                                   key_padding_mask=key_padding_mask,
        #                                   need_weights=False)[0]

        # 应用缩放因子以增加数值稳定性
        if use_scale_factor:
            scale_factor = torch.tensor(self.embed_dims ** -0.5, device=query.device)
            query = query * scale_factor

        # 计算多头注意力
        attn_output, attn_output_weights = self.multihead_attn(
            query=query, key=key, value=value,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )
        # 打印注意力权重的统计信息以帮助调试
        if debug_mode:
            print("Attention Weights Stats - Min: ", attn_output_weights.min().item(),
                " Max: ", attn_output_weights.max().item(),
                " Mean: ", attn_output_weights.mean().item())

            # 检查是否有极端值
            if attn_output_weights.min().item() < -1e10 or attn_output_weights.max().item() > 1e10:
                print("Warning: Extreme values in attention weights detected")
        # 检查注意力权重中是否有 nan 值
        assert not torch.isnan(attn_output_weights).any(), "attn_output_weights contains nan"

        # 检查注意力输出中是否有 nan 值
        assert not torch.isnan(attn_output).any(), "attn_output contains nan"

        # 加入残差连接
        attn_output = residual + self.dropout(attn_output)

        # attn_output = self.dropout(attn_output)


        # 使用 LayerNorm 来稳定训练
        output = self.norm(attn_output)

        # output = residual + self.norm(attn_output)

        return output

class MaskedCrossAttentionV2(nn.Module):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): The number of heads in multihead attention
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 dropout=0.0,
                 init_cfg=None,
                 batch_first=True,
                 **kwargs
                 ):
        super(MaskedCrossAttentionV2, self).__init__()

        self.init_cfg = init_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.multihead_attn = nn.MultiheadAttention(embed_dims,
                                                    num_heads,
                                                    dropout=dropout,
                                                    batch_first=batch_first,
                                                    **kwargs)

        # Add LayerNorm layers
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                spatial_shapes=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (bs, max_num_vectors, embed_dims)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            spatial_shapes (Tensor): number of polylines in each batch.
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if key is None:
            key = value

        if residual is None:
            residual = query

        if query_pos is not None:
            query = query + query_pos

        bs, _, _ = query.shape
        key_padding_mask = torch.arange(spatial_shapes.max(), device=spatial_shapes.device).unsqueeze(0).repeat(bs, 1) >= spatial_shapes.unsqueeze(1)

        # Normalize the query before attention
        query = self.norm1(query)

        queries = self.multihead_attn(query=query, key=key, value=value,
                                      key_padding_mask=key_padding_mask, need_weights=False)[0]

        # Residual connection and normalization
        queries = self.norm2(queries + residual)

        return queries

class GridPrior(nn.Module):
    def __init__(self, disp_input_dim=40, cell_input_dim=5, state_input_dim=2):
        super().__init__()

        # 定义三个卷积层，卷积核大小为1x1，通道数变为32
        self.conv_disp = nn.Conv2d(in_channels=disp_input_dim, out_channels=32, kernel_size=1)
        self.conv_cell = nn.Conv2d(in_channels=cell_input_dim, out_channels=32, kernel_size=1)
        self.conv_state = nn.Conv2d(in_channels=state_input_dim, out_channels=32, kernel_size=1)

        # 对应的BatchNorm和ReLU
        self.bn_disp = nn.BatchNorm2d(32)
        self.bn_cell = nn.BatchNorm2d(32)
        self.bn_state = nn.BatchNorm2d(32)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, all_disp_field_gt, cell_input, state_input):
        """
        Args:
            all_disp_field_gt: [bs, disp_input_dim, bev_w, bev_h]
            cell_input: [bs, cell_input_dim, bev_w, bev_h]
            state_input: [bs, state_input_dim, bev_w, bev_h]

        Returns:
            disp_feat: [bs, 32, bev_w, bev_h] 提取后的位移特征
            cell_feat: [bs, 32, bev_w, bev_h] 提取后的细胞特征
            state_feat: [bs, 32, bev_w, bev_h] 提取后的状态特征
        """

        # 对位移特征进行卷积 -> BN -> ReLU
        disp_feat = self.conv_disp(all_disp_field_gt)  # [bs, 32, bev_w, bev_h]
        disp_feat = self.bn_disp(disp_feat)
        disp_feat = self.relu(disp_feat)

        # 对细胞特征进行卷积 -> BN -> ReLU
        cell_feat = self.conv_cell(cell_input)         # [bs, 32, bev_w, bev_h]
        cell_feat = self.bn_cell(cell_feat)
        cell_feat = self.relu(cell_feat)

        # 对状态特征进行卷积 -> BN -> ReLU
        state_feat = self.conv_state(state_input)      # [bs, 32, bev_w, bev_h]
        state_feat = self.bn_state(state_feat)
        state_feat = self.relu(state_feat)

        # 返回三个特征提取后的结果
        return disp_feat, cell_feat, state_feat

class GridPriorv2(nn.Module):
    def __init__(self,
                 seq_len=20,
                 cell_input_dim=16,
                 num_cls=5,
                 num_state=2,
                 dropout=0.1,
                 temperature=0.5):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        # ==================== 局部特征提取 ====================
        # 原始特征投影层
        self.raw_disp_proj = nn.Conv3d(2, cell_input_dim//4, 1)  # [B,2,T,H,W]->[B,D/4,T,H,W]
        self.raw_cls_proj = nn.Conv2d(num_cls, cell_input_dim//8, 1)  # [B,C,H,W]->[B,D/8,H,W]
        self.raw_state_proj = nn.Conv2d(num_state, cell_input_dim//8, 1)  # [B,S,H,W]->[B,D/8,H,W]

        # 位移编码器（3D CNN保持维度）
        self.disp_encoder = nn.Sequential(
            nn.Conv3d(2, cell_input_dim, (3,3,3), padding=(1,1,1)),
            nn.InstanceNorm3d(cell_input_dim),
            nn.GELU(),
            nn.Conv3d(cell_input_dim, cell_input_dim, (3,3,3), padding=(1,1,1))
        )

        # 类别编码器（2D CNN保持维度）
        self.cls_encoder = nn.Sequential(
            nn.Conv2d(num_cls, cell_input_dim//2, 3, padding=1),
            nn.BatchNorm2d(cell_input_dim//2),
            nn.GELU(),
            nn.Conv2d(cell_input_dim//2, cell_input_dim//2, 3, padding=1)
        )

        # 状态编码器（2D CNN保持维度）
        self.state_encoder = nn.Sequential(
            nn.Conv2d(num_state, cell_input_dim//2, 3, padding=1),
            nn.BatchNorm2d(cell_input_dim//2),
            nn.GELU(),
            nn.Conv2d(cell_input_dim//2, cell_input_dim//2, 3, padding=1)
        )

        # =================== 全局特征提取 ====================
        # # QKV生成层（包含残差连接）
        # self.qkv_gen = nn.Sequential(
        #     nn.Conv2d( (cell_input_dim + cell_input_dim//4 )*2,  # 增加原始特征通道
        #               cell_input_dim//2, 1),
        #     nn.GELU(),
        #     nn.Conv2d(cell_input_dim//2, cell_input_dim//8*3, 1)  # 输出Q/K/V各占1/4
        # )
        self.CAlayer = CAlayer(channel=(cell_input_dim + cell_input_dim//4 )*2,
                               h=256, w=256)

        # 注意力后的重建层
        self.recon_conv = nn.Sequential(
            nn.Conv2d(cell_input_dim//8, cell_input_dim, 3, padding=1),
            nn.GroupNorm(4, cell_input_dim),
            nn.GELU()
        )

        # ==================== 融合参数 ======================
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, disp_gt, cls_gt, state_gt):
        """输入维度保持不变"""
        B, T, H, W, _ = disp_gt.shape

        # ============== 局部特征提取 ================
        # 处理位移特征
        disp_feat = self.disp_encoder(disp_gt.permute(0,4,1,2,3))  # [B,D,T,H,W]
        avg_disp = disp_feat.mean(dim=2)  # [B,D,H,W]

        # 处理分类和状态特征
        cls_feat = self.cls_encoder(cls_gt)
        state_feat = self.state_encoder(state_gt)

        # ============= 原始特征投影 ================
        # 位移原始特征投影（时间维度平均）
        raw_disp = self.raw_disp_proj(disp_gt.permute(0,4,1,2,3)).mean(dim=2)  # [B,D/4,H,W]
        raw_cls = self.raw_cls_proj(cls_gt)  # [B,D/8,H,W]
        raw_state = self.raw_state_proj(state_gt)  # [B,D/8,H,W]

        # ============= 特征合并 ====================
        combined_feat = torch.cat([
            avg_disp,
            cls_feat,
            state_feat,
            raw_disp,
            raw_cls,
            raw_state
        ], dim=1)[:, :, :H, :W]  # [B, D+D/2+D/2 + D/4 + D/8*2, H, W]
        # breakpoint()
        # # ============= 全局注意力计算 ===============

        # # 生成QKV
        # qkv = self.qkv_gen(combined_feat)  # [B, 3*(D//4), H, W]
        # Q, K, V = torch.chunk(qkv, 3, dim=1)  # 各通道数为D//4

        # # 维度重组
        # Q_seq = Q.view(B, -1, H*W).permute(0,2,1)  # [B, HW, C]
        # K_seq = K.view(B, -1, H*W).permute(0,2,1)
        # V_seq = V.view(B, -1, H*W).permute(0,2,1)

        # # 高效注意力计算
        # # attn = F.scaled_dot_product_attention(Q_seq, K_seq, V_seq)
        # # 计算注意力权重
        # attn = torch.matmul(Q_seq / self.temperature, K_seq.transpose(1, 2))

        # # 应用softmax函数得到最终的注意力分布
        # attn = self.dropout(F.softmax(attn, dim=-1))

        # # 使用注意力分布加权求和得到输出
        # output = torch.matmul(attn, V_seq)

        # attn_feat = output.permute(0,2,1).view(B, -1, H, W)

        # # 特征重建
        # global_feat = self.recon_conv(attn_feat)
        global_feat = self.CAlayer(combined_feat)
        # ============= 自适应融合 ================
        fusion_gate = torch.sigmoid(self.alpha)
        fused = fusion_gate * combined_feat + (1-fusion_gate) * global_feat
        # fused = fusion_gate * combined_feat

        return fused

if __name__ == "__main__":

    # 假设输入的尺寸
    bs, bev_w, bev_h = 8, 64, 64
    disp_input_dim, cell_input_dim, state_input_dim = 40, 5, 2

    # 创建模型
    model = GridPrior(disp_input_dim=disp_input_dim, cell_input_dim=cell_input_dim, state_input_dim=state_input_dim)

    # 创建假数据
    all_disp_field_gt = torch.randn(bs, disp_input_dim, bev_w, bev_h)
    cell_input = torch.randn(bs, cell_input_dim, bev_w, bev_h)
    state_input = torch.randn(bs, state_input_dim, bev_w, bev_h)

    # 前向传播
    disp_feat, cell_feat, state_feat = model(all_disp_field_gt, cell_input, state_input)

    # 打印输出形状
    print(f"disp_feat shape: {disp_feat.shape}")  # 应该是 [bs, 32, bev_w, bev_h]
    print(f"cell_feat shape: {cell_feat.shape}")  # 应该是 [bs, 32, bev_w, bev_h]
    print(f"state_feat shape: {state_feat.shape}")  # 应该是 [bs, 32, bev_w, bev_h]

    """
    pos_encoder=dict(
            type='SineContinuousPositionalEncoding',
            num_feats=16,  # 2 * 16 = 32 final dim
            temperature=1000,
            normalize=True,
            range=[point_cloud_range[3] - point_cloud_range[0], point_cloud_range[4] - point_cloud_range[1]],
            offset=[point_cloud_range[0], point_cloud_range[1]],
        ),
    """
    ins_pe = SineContinuousPositionalEncoding(
        num_feats=16,  # 2 * 16 = 32 final dim
        temperature=1000,
        normalize=False,
        range=[100, 100],
        offset=[0, 0],
    )
    bs = 4
    num_grid = 100
    num_ins = 10
    cat_num = 5
    seq_len = 20
    # tensor: [bs, num_ins * num_grid, 2]
    test_tensor = torch.randn(bs, num_ins, num_grid, 2)

    # disp tensor
    test_disp_tensor = torch.randn(bs, num_ins, num_grid, 2)

    test_seq_disp_tensor = torch.randn(bs, seq_len, num_ins, num_grid, 2)

    # onehot tensor
    test_tensor_cat = torch.randn(bs, num_ins, cat_num)
    # tensor_after_pe = ins_pe(test_tensor)
    # print(tensor_after_pe.shape)

    ins_pe_list = [test_tensor[i, ...] for i in range(test_tensor.shape[0])]
    onehot_category_list = [test_tensor_cat[i, ...] for i in range(test_tensor_cat.shape[0])]
    ins_disp_list = [test_disp_tensor[i,...] for i in range(test_disp_tensor.shape[0])]
    """
     map_encoder=dict(
        type='MapGraphTransformer',
        input_dim=360,  # 32 * 11 + 8
        dmodel=_dim_,      # TODO: maybe change?
        hidden_dim=_dim_,  # TODO: maybe change?
        nheads=_num_heads_,
        nlayers=6,
        batch_first=True,
        pos_encoder=dict(
            type='SineContinuousPositionalEncoding',
            num_feats=16,  # 2 * 16 = 32 final dim
            temperature=1000,
            normalize=True,
            range=[point_cloud_range[3] - point_cloud_range[0], point_cloud_range[4] - point_cloud_range[1]],
            offset=[point_cloud_range[0], point_cloud_range[1]],
        ),
    ),"""

    ins_encoder = GridInsTransformer(
        input_dim=32*num_grid + 32*num_grid + cat_num,  # 32 * 11 + 8
        dmodel=32,      # TODO: maybe change?
        hidden_dim=2048,  # TODO: maybe change?
        nheads=8,
        nlayers=6,
        batch_first=True,
        pos_encoder="sin",
        disp_encoder="mlp",
    )
    ins_feat = ins_encoder(ins_pe_list, ins_disp_list, onehot_category_list)
    print(ins_feat[0].shape)

    """
    if map_graph_feats is not None:
            if isinstance(map_graph_feats, list):
                # batch, num_polylines * feat_dim
                map_graph_shapes = torch.tensor([mfeat.shape[0] for mfeat in map_graph_feats],
                                                device=map_graph_feats[0].device)

                max_num_polylines = map_graph_shapes.max()
                map_graph_feats = torch.stack([
                    torch.cat([mfeat, mfeat.new_zeros(max_num_polylines - mfeat.shape[0],
                                                      mfeat.shape[1])], dim=0) for mfeat in map_graph_feats
                ])  # batch, max_num_polylines, feat_dim
            else:
                raise NotImplementedError()

            # map_graph_flat = torch.cat(map_graph_feats, dim=0)  # (batch total num_polylines , feat_dim) ~= (batch * num_polylines, feat_dim)
        else:
            map_graph_shapes = None
            map_graph_feats = None
    """
    if isinstance(ins_feat, list):
        # batch, num_polylines * feat_dim
        ins_graph_shapes = torch.tensor([mfeat.shape[0] for mfeat in ins_feat],
                                        device=ins_feat[0].device)

        max_num_polylines = ins_graph_shapes.max()
        ins_feat = torch.stack([
            torch.cat([mfeat, mfeat.new_zeros(max_num_polylines - mfeat.shape[0],
                                              mfeat.shape[1])], dim=0) for mfeat in ins_feat
        ])  # batch, max_num_polylines, feat_dim

    print(ins_feat.shape)

    bev_cross_attn_layer = MaskedCrossAttention(embed_dims=32,
                 num_heads=8,
                 dropout=0.0,
                 init_cfg=None,
                 batch_first=True)

    """
     query = self.attentions[attn_index](
                    query,
                    None,
                    map_graph_feats,
                    residual=query,
                    query_pos=bev_pos,
                    spatial_shapes=map_graph_shapes,
                    **kwargs)
                attn_index += 1
                identity = query"""
    # bev tokens
    bev_dim = 32
    bev_h, bev_w = 256, 256
    simple_pe = SimplePositionEncoding(num_feats=16, row_num_embed=bev_h, col_num_embed=bev_w)

    bev = torch.randn(bs, bev_dim, bev_w, bev_h)
    bev_mask = torch.zeros((bs, bev_h, bev_w))
    bev_pos = simple_pe(bev_mask)
    # breakpoint()
    bev_pos = bev_pos.flatten(2).permute(2, 0, 1).permute(1, 0, 2)
    bev = bev.view(-1, bev_dim, bev_w*bev_h).permute(0, 2, 1)
    bev_feat = bev_cross_attn_layer(bev, ins_feat, ins_feat, query_pos=bev_pos, residual=bev, spatial_shapes=ins_graph_shapes)
    print(bev_feat.shape)
