import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import SElayer, SelfAttention
from model.prior_extractor import SimplePositionEncoding
class MaxSigmoid(nn.Module):
    def __init__(self, bev_embed_dim=32, ins_embed_dim=32):
        super(MaxSigmoid, self).__init__()

    def forward(self, bev_features, instance_masks):
        """
        根据给定公式更新BEV特征图。

        参数:
            bev_features (torch.Tensor): 形状为 [bs, dim, h, w] 的BEV特征图。
            instance_masks (torch.Tensor): 形状为 [bs, seq, dim] 的实例掩码嵌入。

        返回:
            torch.Tensor: 更新后的BEV特征图 形状为 [bs, dim, h, w]。
        """
        bs, dim, h, w = bev_features.shape
        _, seq, _ = instance_masks.shape

        # 将bev_features展平为 [bs, dim, h*w]
        bev_features_flat = bev_features.view(bs, dim, -1)

        # 调整instance_masks形状以匹配矩阵乘法需求 [bs, dim, seq]
        instance_masks_t = instance_masks.transpose(1, 2)  # 形状为 [bs, dim, seq]

        # 计算 Hadamard 乘积
        # 首先计算bev_features与instance_masks的点积，得到[bs, h*w, seq]
        hadamard_product = torch.matmul(bev_features_flat.transpose(1, 2), instance_masks_t)  # 形状为 [bs, h*w, seq]

        # 对seq维度求最大值，结果形状为 [bs, h*w]
        max_hadamard, _ = torch.max(hadamard_product, dim=2)  # 形状为 [bs, h*w]

        # 应用 sigmoid 函数
        sigmoid_max = torch.sigmoid(max_hadamard).unsqueeze(1)  # 形状为 [bs, 1, h*w]

        # 更新 BEV 特征向量 F'
        updated_bev_features = bev_features_flat * sigmoid_max

        # 将更新后的特征图重新调整回原始形状 [bs, dim, h, w]
        updated_bev_features = updated_bev_features.view(bs, dim, h, w)

        return updated_bev_features

class EarlyFusion(nn.Module):
    def __init__(self, bev_embed_dim=32, ins_embed_dim=32):
        super(EarlyFusion, self).__init__()

        self.SElayer = SElayer(channel=bev_embed_dim)
        self.MSA = SelfAttention(embed_size=ins_embed_dim, heads=4)

        self.MAX_Sig = MaxSigmoid(bev_embed_dim=bev_embed_dim, ins_embed_dim=ins_embed_dim)

    def forward(self, bev, ins_query):
        """
            bev: [bs, dim, h, w]
            ins_query: [bs, seq, dim]

        """
        bev = self.SElayer(bev)
        ins_query = self.MSA(ins_query, ins_query, ins_query)

        early_fusion_output = self.MAX_Sig(bev, ins_query)

        return early_fusion_output, ins_query


class LateFusion(nn.Module):
    def __init__(self, bev_embed_dim=32, ins_embed_dim=32, output_dim=32, heads=8, position_embedding=True):
        super(LateFusion, self).__init__()
        self.bev_embed_dim = bev_embed_dim
        self.ins_embed_dim = ins_embed_dim
        self.heads = heads
        self.position_embedding = position_embedding

        # 线性变换层，用于生成Q, K, V
        self.query_linear = nn.Linear(bev_embed_dim, output_dim)
        self.key_linear = nn.Linear(ins_embed_dim, output_dim)
        self.value_linear = nn.Linear(ins_embed_dim, output_dim)

        # 输出线性层
        self.fc_out = nn.Linear(output_dim, output_dim)

        # if position_embedding:
            # 初始化位置嵌入参数
            # self.position_embeddings = nn.Parameter(torch.randn(1, 1, *bev_embed_dim))  # [1, 1, dim]


    def add_positional_encoding(self, bev_features):
        """
        添加位置嵌入到BEV特征图中。

        参数:
            bev_features (torch.Tensor): 形状为 [bs, dim, h, w] 的BEV特征图。

        返回:
            torch.Tensor: 添加了位置嵌入的BEV特征图 形状仍为 [bs, dim, h, w]。
        """
        bs, dim, h, w = bev_features.shape
        self.simple_pe = SimplePositionEncoding(num_feats=16, row_num_embed=h, col_num_embed=w)
        # if not hasattr(self, 'position_embeddings'):
        #     raise ValueError("Position embedding is not enabled or initialized.")

        # 生成位置嵌入，尺寸与输入相同
        # position_encodings = self.position_embeddings.expand(bs, dim, h, w)
        position_encodings = self.simple_pe(bev_features)

        return bev_features + position_encodings

    def forward(self, bev_features, ins_query_features):
        """
        参数:
            bev_features (torch.Tensor): 形状为 [bs, dim, h, w] 的BEV特征图。
            ins_query_features (torch.Tensor): 形状为 [bs, seq, dim] 的实例查询特征。

        返回:
            torch.Tensor: 融合后的BEV特征图 形状为 [bs, output_dim, h, w]。
        """
        if self.position_embedding:
            bev_features = self.add_positional_encoding(bev_features)

        bs, dim, h, w = bev_features.shape
        _, seq, _ = ins_query_features.shape

        # 将BEV特征图展平为 [bs, dim, h*w] 并转置为 [bs, h*w, dim]
        bev_features_flat = bev_features.view(bs, dim, -1).transpose(1, 2)  # 形状为 [bs, h*w, dim]

        # 使用线性变换生成Q, K, V
        Q = self.query_linear(bev_features_flat)  # 形状为 [bs, h*w, output_dim]
        K = self.key_linear(ins_query_features)   # 形状为 [bs, seq, output_dim]
        V = self.value_linear(ins_query_features) # 形状为 [bs, seq, output_dim]

        # 计算注意力得分
        energy = torch.bmm(Q, K.transpose(1, 2)) / (self.ins_embed_dim ** 0.5)  # 形状为 [bs, h*w, seq]

        # 计算注意力权重
        attention_weights = F.softmax(energy, dim=-1)  # 形状为 [bs, h*w, seq]

        # 应用注意力权重到值上
        attended_values = torch.bmm(attention_weights, V)  # 形状为 [bs, h*w, output_dim]

        # 通过最终的线性层
        out = self.fc_out(attended_values)  # 形状为 [bs, h*w, output_dim]

        # 恢复原始形状 [bs, output_dim, h, w]
        out = out.transpose(1, 2).view(bs, -1, h, w)

        return out
