import torch.nn.functional as F
import torch.nn as nn
import torch,math

# modified by SE model
class SElayer(nn.Module):
    """
    reduction is the hyper-parameter
    """
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# activate function in CAlayer
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CAlayer(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CAlayer, self).__init__()

        self.h = h
        self.w = w

        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)

        self.relu = nn.ReLU()
        # ReLU can be replaced by h_swish
        self.act = h_swish()

        self.bn = nn.BatchNorm2d(channel//reduction)

        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x): # torch.size(2, 64, 5, 5)
        bs, c, h, w = x.size()

        # (b, c, h, w) -> (b, c, h, 1) -> (b, c, 1, h)
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)

        # (b, c, h, w) -> (b, c, 1, w)
        x_w = self.avg_pool_y(x)

        # cat and conv
        # (b, c, 1, w) cat (b, c, 1, h) -> (b, c, 1, w + h)
        # (b, c, 1, w + h) -> (b, c/r, 1, w + h)
        x_cat_conv_relu = self.act(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        # split
        # (b, c/r, 1, w + h) -> (b, c/r, 1, h) + (b, c/r, 1, w)
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)

        # (b, c/r, 1, h) -> (b, c, h, 1)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))

        # (b, c/r, 1, w) -> (b, c, 1, w)
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        # out = x * s_h.expand(x.shape) * s_w.expand_as(x.shape)

        return out

class ECAlayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1):  #64
        super(ECAlayer, self).__init__()
        kernel_size = int(abs((math.log(channel,2)+  b)/gamma))  #3
        kernel_size = kernel_size if kernel_size % 2  else kernel_size+1  #3
        padding = kernel_size//2
        self.avg_pool =nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()  #torch.Size([2, 64, 26, 26])
        #变成序列的形式
        avg = self.avg_pool(x).view([b, 1, c])   #torch.Size([2, 64, 1, 1])   torch.Size([2, 1, 64])
        out = self.conv(avg)                     #torch.Size([2, 1, 64])
        out = self.sigmoid(out).view([b, c, 1, 1])  #torch.Size([2, 64, 1, 1])
        return  out * x

# channel attention
class channel_attention(nn.Module):
    def __init__(self, channel, ration=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//ration, bias=False),
            nn.ReLU(),
            nn.Linear(channel//ration, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):           #torch.Size([2, 64, 5, 5])
        b, c, h, w = x.size()
        avg_pool = self.avg_pool(x).view([b, c])  #torch.Size([2, 64])
        max_pool = self.max_pool(x).view([b, c])  #torch.Size([2, 64])

        avg_fc = self.fc(avg_pool)  #torch.Size([2, 64])
        max_fc = self.fc(max_pool)  #torch.Size([2, 64])

        out = self.sigmoid(max_fc+avg_fc).view([b, c, 1, 1])  ##torch.Size([2, 64, 1, 1])
        return x * out

# spatial attention
class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # channel maxPooling
        max_pool = torch.max(x, dim=1, keepdim=True).values  #torch.Size([2, 1, 5, 5])
        avg_pool = torch.mean(x, dim=1, keepdim=True)        #torch.Size([2, 1, 5, 5])
        pool_out = torch.cat([max_pool, avg_pool], dim=1)    #torch.Size([2, 2, 5, 5])
        conv = self.conv(pool_out)                           #torch.Size([2, 1, 5, 5])
        out = self.sigmoid(conv)

        return out * x

# combine channel and spatial
class CBAM(nn.Module):
    def __init__(self, channel, ration=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = channel_attention(channel, ration)
        self.spatial_attention = spatial_attention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x)    #torch.Size([2, 64, 5, 5])
        out = self.spatial_attention(out)  #torch.Size([2, 64, 5, 5])

        return out



class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        """
        初始化自注意力层。

        参数:
            embed_size (int): 输入特征的维度大小。
            heads (int): 注意力头的数量。为了简化 我们假设embed_size可以被heads整除。
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 定义线性变换层，用于生成Q, K, V
        self.values = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size*heads, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 将输入分割成多个头
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # 通过线性层传递
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 计算注意力得分
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # breakpoint()
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size * self.heads
        )

        out = self.fc_out(out)
        return out


import math
import torch
from torch import nn
#from torchstat import stat  # 查看网络参数

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out




# class Attention(nn.Module):
    # def __init__(self,channel,b=1, gamma=2):
        # super(Attention, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        # #一维卷积
        # t = int(abs((math.log(channel, 2) + b) / gamma))
        # k = t if t % 2 else t + 1
        # self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        # self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        # self.sigmoid = nn.Sigmoid()
        # self.mix = Mix()
        # #全连接
        # #self.fc = nn.Linear(channel,channel)
        # #self.softmax = nn.Softmax(dim=1)

    # def forward(self, input):
        # x = self.avg_pool(input)
        # x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # out1 = self.sigmoid(x1)
        # out2 = self.fc(x)
        # out2 = self.sigmoid(out2)
        # out = self.mix(out1,out2)
        # out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # #out = self.softmax(out)

        # return input*out
class GlobalLocalAttention(nn.Module):
    def __init__(self,channel,b=1, gamma=2):
        super(GlobalLocalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        #一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()


    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)#(1,64,1)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)#(1,1,64)
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)#(1,64,1,1)
        #x1 = x1.transpose(-1, -2).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)

        #out2 = self.fc(x)
        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)

        return input*out

if __name__ == '__main__':
    input = torch.rand(1,32,256,256)

    A = GlobalLocalAttention(channel=32)
    #stat(A, input_size=[64, 1, 1])
    y = A(input)
    print(y.size())
