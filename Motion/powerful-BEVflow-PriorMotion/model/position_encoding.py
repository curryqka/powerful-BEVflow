import math
import torch
from torch import nn

class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return self.pe[:, :, :x.size(2), :x.size(3)]


class PositionEncodingSinev2(nn.Module):

    def __init__(self, d_model, temperature=10000, normalize=False, scale=None):
        super().__init__()

        self.d_model = d_model
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale if passed!')

        if scale is None:
            scale = 2 * math.pi

        self.scale = scale

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        masks = torch.zeros([x.shape[0], x.shape[2], x.shape[3]], dtype=torch.bool, device=x.device)
        not_mask = ~masks

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.d_model, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.d_model)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class RoPEPositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), npe=None, ropefp16=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        i_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(-1) # [H, 1]
        j_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(-1) # [W, 1]

        assert npe is not None
        train_res_H, train_res_W, test_res_H, test_res_W = npe[0], npe[1], npe[2], npe[3] # train_res_H, train_res_W, test_res_H, test_res_W
        i_position, j_position = i_position * train_res_H / test_res_H, j_position * train_res_W / test_res_W

        div_term = torch.exp(torch.arange(0, d_model//4, 1).float() * (-math.log(10000.0) / (d_model//4)))
        div_term = div_term[None, None, :]  # [1, 1, C//4]

        sin = torch.zeros(*max_shape, d_model//2, dtype=torch.float16 if ropefp16 else torch.float32)
        cos = torch.zeros(*max_shape, d_model//2, dtype=torch.float16 if ropefp16 else torch.float32)
        sin[:, :, 0::2] = torch.sin(i_position * div_term).half() if ropefp16 else torch.sin(i_position * div_term)
        sin[:, :, 1::2] = torch.sin(j_position * div_term).half() if ropefp16 else torch.sin(j_position * div_term)
        cos[:, :, 0::2] = torch.cos(i_position * div_term).half() if ropefp16 else torch.cos(i_position * div_term)
        cos[:, :, 1::2] = torch.cos(j_position * div_term).half() if ropefp16 else torch.cos(j_position * div_term)

        sin = sin.repeat_interleave(2, dim=-1)
        cos = cos.repeat_interleave(2, dim=-1)

        self.register_buffer('sin', sin.unsqueeze(0), persistent=False)  # [1, H, W, C//2]
        self.register_buffer('cos', cos.unsqueeze(0), persistent=False)  # [1, H, W, C//2]

    def forward(self, x, ratio=1):
        """
        Args:
            x: [N, H, W, C]
        """
        x = x.permute(0, 2, 3, 1)
        x = (x * self.cos[:, :x.size(1), :x.size(2), :]) + (self.rotate_half(x) * self.sin[:, :x.size(1), :x.size(2), :])
        return x.permute(0, 3, 1, 2)

    def rotate_half(self, x):
        x = x.unflatten(-1, (-1, 2))
        x1, x2 = x.unbind(dim=-1)
        return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device) # w
        j = torch.arange(h, device=x.device) # h
        # nn.Embedding
        # torch.nn.Embedding(num_embeddings, embedding_dim)
        # num_embeddings (python:int) —— 词典的大小尺寸，比如总共出现 5000 个词，那就输入 5000 。
        # 此时index 为（ 0 − 4999 ）
        # embedding_dim (python:int)–—— 嵌入向量的维度，即用多少维来表示一个符号。
        # 输入必须是 LongTensor，FloatTensor 需通过 tensor.long() 方法转成 LongTensor。
        # 经过实验 num_embeddings.weight 均为随机产生的。
        # 输出结果含有 requires_grad ，说明此步骤在反向传播中需要计算。
        x_emb = self.col_embed(i) # [w,256]
        y_emb = self.row_embed(j) # [h,256]
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),  # h,w,256
            y_emb.unsqueeze(1).repeat(1, w, 1),  # h,w,256
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        # cat 之后是 h,w,512   permute: 512,h,w  unsqueeze 1,512,h,w  b,512,h,w
        return pos
