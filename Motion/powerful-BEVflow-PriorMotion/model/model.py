# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from model.attention import SElayer
from model.vae_model import SimpleVAE
from model.transformer import Transformer, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from model.position_encoding import PositionEncodingSinev2


class CellClassification(nn.Module):
    def __init__(self, category_num=5, in_channel=32):
        super(CellClassification, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, category_num, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        # output: [bs, 5, h, w]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class StateEstimation(nn.Module):
    def __init__(self, motion_category_num=2, in_channel=32):
        super(StateEstimation, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, motion_category_num, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        # output: [bs, 2, h, w]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class MotionPrediction(nn.Module):
    def __init__(self, seq_len, in_channel=32):
        super(MotionPrediction, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 2 * seq_len, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        # output: [bs, 2 * seq, h, w]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class Conv3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        # input x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq_len, h, w)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq_len, c, h, w)

        return x

class VelocityEncoder(nn.Module):
    def __init__(self, velocity_feat_size = 1, with_direction=False, with_cos=False):
        super(VelocityEncoder, self).__init__()
        self.input_dim = velocity_feat_size
        if with_direction:
            self.input_dim += 1
        elif with_cos:
            self.input_dim+= 3
        self.conv3d_1 = Conv3D(self.input_dim, 16, kernel_size=(3, 1 ,1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(16, 32, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

    def forward(self, x):
        batch, seq, h, w, dim = x.size()
        x = x.view(-1, seq, dim, h, w)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        return x

class VelocityEncoderV2(nn.Module):
    def __init__(self, velocity_feat_size = 1, with_direction=False, with_cos=False):
        super(VelocityEncoderV2, self).__init__()
        self.input_dim = velocity_feat_size
        if with_direction:
            self.input_dim += 1
        elif with_cos:
            self.input_dim+= 3

        self.backbone = STPN(self.input_dim)

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        x = self.backbone(bevs)

        return x


class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


class STPN(nn.Module):
    def __init__(self, height_feat_size=13):
        super(STPN, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))

        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = F.adaptive_max_pool3d(x_2, (1, None, None))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.adaptive_max_pool3d(x, (1, None, None))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))
        # [bs, 32, h, w]

        return res_x

class STPNv2(nn.Module):
    def __init__(self, height_feat_size=13):
        super(STPNv2, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(3, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))

        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = F.adaptive_max_pool3d(x_2, (1, None, None))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.adaptive_max_pool3d(x, (1, None, None))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))
        # [bs, 32, h, w]

        return x_4, res_x

class LOWERSTPN(nn.Module):
    def __init__(self, height_feat_size=13):
        super(LOWERSTPN, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))
        self.conv3d_2 = Conv3D(128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0))

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

    def forward(self, x):
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))

        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(self.bn5_1(self.conv5_1(torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1))))
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = F.adaptive_max_pool3d(x_2, (1, None, None))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_6 = F.relu(self.bn6_1(self.conv6_1(torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1))))
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = F.adaptive_max_pool3d(x_1, (1, None, None))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(self.bn7_1(self.conv7_1(torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1))))
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = F.adaptive_max_pool3d(x, (1, None, None))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(self.bn8_1(self.conv8_1(torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1))))
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        return res_x


class MotionNet(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13):
        super(MotionNet, self).__init__()
        self.out_seq_len = out_seq_len

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)
        self.stpn = STPN(height_feat_size=height_feat_size)

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # Backbone network
        x = self.stpn(bevs)

        # Cell Classification head
        cell_class_pred = self.cell_classify(x)

        # Motion State Classification head
        state_class_pred = self.state_classify(x)

        # Motion Displacement prediction
        disp = self.motion_pred(x)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))

        return disp, cell_class_pred, state_class_pred, x


class InsEncoder(nn.Module):
    def __init__(self, embed_channel=32, ins_query_num=95, depth_decoder=2, depth_encoder=0):
        super(InsEncoder, self).__init__()

        self.d_model = embed_channel

        self.query_num = ins_query_num

        self.query_embed = torch.nn.Embedding(ins_query_num, embed_channel)

        # self.decoder = Transformer(d_model=embed_channel, nhead=8, num_encoder_layers=2,
        #          num_decoder_layers=2, dim_feedforward=64, dropout=0.1,
        #          activation="relu", normalize_before=False,
        #          return_intermediate_dec=False)

        self.depth_encoder = depth_encoder
        if depth_encoder > 0:
            encoder_layer = TransformerEncoderLayer(d_model=embed_channel, nhead=8, dim_feedforward=128,
                                                dropout=0.1, activation="relu", normalize_before=False)
            self.encoder = TransformerEncoder(encoder_layer, depth_encoder, None)

        decoder_layer = TransformerDecoderLayer(d_model=embed_channel, nhead=8, dim_feedforward=128,
                                                dropout=0.1, activation="relu", normalize_before=False)
        decoder_norm = nn.LayerNorm(embed_channel)
        self.decoder = TransformerDecoder(decoder_layer, depth_decoder, decoder_norm,
                                          return_intermediate=False)

        self.mlp = nn.Sequential(
            nn.Linear(embed_channel, embed_channel),
            nn.ReLU(inplace=True),
            nn.Linear(embed_channel, embed_channel)
        )

        self.pe_sine = PositionEncodingSinev2(embed_channel // 2)

        self.conv = nn.Sequential(
            nn.Conv2d(embed_channel, embed_channel, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channel, embed_channel, 1, 1, 0)
        )

    def forward(self, x, x0=None):
        """
            x: small size of bev: [bs, c, h/4, w/4]
            x0: raw_bev: [bs, c, h, w]
        """
        B = x.size(0)

        pos = self.pe_sine(x).flatten(2).permute(2, 0 ,1)

        tgt = torch.zeros((B, 1, self.d_model)).to(x.device).repeat(1, self.query_num, 1).permute(1, 0, 2)

        memory = x.flatten(2).permute(2, 0, 1)

        if self.depth_encoder > 0:
            memory = self.encoder(memory, pos=pos)

        hs = self.decoder(tgt, memory, pos=pos, query_pos=self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)).transpose(1, 2)

        # hs, memory = self.decoder(x, tgt, pos, self.query_embed.weight)

        embed = self.mlp(hs[0])

        x = self.conv(x)

        mask = torch.einsum('bnc,bchw->bnhw', embed, x)

        return mask, hs[0]

class InsEncoderV2(nn.Module):
    def __init__(self, embed_channel=32, ins_query_num=95, depth_decoder=2, depth_encoder=0):
        super(InsEncoderV2, self).__init__()

        self.d_model = embed_channel

        self.query_num = ins_query_num

        self.query_embed = torch.nn.Embedding(ins_query_num, embed_channel)

        # self.decoder = Transformer(d_model=embed_channel, nhead=8, num_encoder_layers=2,
        #          num_decoder_layers=2, dim_feedforward=64, dropout=0.1,
        #          activation="relu", normalize_before=False,
        #          return_intermediate_dec=False)

        self.depth_encoder = depth_encoder
        if depth_encoder > 0:
            encoder_layer = TransformerEncoderLayer(d_model=embed_channel, nhead=8, dim_feedforward=128,
                                                dropout=0.1, activation="relu", normalize_before=False)
            self.encoder = TransformerEncoder(encoder_layer, depth_encoder, None)

        decoder_layer = TransformerDecoderLayer(d_model=embed_channel, nhead=8, dim_feedforward=128,
                                                dropout=0.1, activation="relu", normalize_before=False)
        decoder_norm = nn.LayerNorm(embed_channel)
        self.decoder = TransformerDecoder(decoder_layer, depth_decoder, decoder_norm,
                                          return_intermediate=False)

        self.mlp = nn.Sequential(
            nn.Linear(embed_channel, embed_channel),
            nn.ReLU(inplace=True),
            nn.Linear(embed_channel, embed_channel)
        )

        self.pe_sine = PositionEncodingSinev2(embed_channel // 2)

        self.conv = nn.Sequential(
            nn.Conv2d(embed_channel, embed_channel, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channel, embed_channel, 1, 1, 0)
        )

    def forward(self, x, x0=None):
        """
            x: small size of bev: [bs, c, h/4, w/4]
            x0: raw_bev: [bs, c, h, w]
            hs: [depth, bs, seq, dim]
        """
        B = x.size(0)

        pos = self.pe_sine(x).flatten(2).permute(2, 0 ,1)

        tgt = torch.zeros((B, 1, self.d_model)).to(x.device).repeat(1, self.query_num, 1).permute(1, 0, 2)

        memory = x.flatten(2).permute(2, 0, 1)

        if self.depth_encoder > 0:
            memory = self.encoder(memory, pos=pos)

        hs = self.decoder(tgt, memory, pos=pos, query_pos=self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)).transpose(1, 2)

        # hs, memory = self.decoder(x, tgt, pos, self.query_embed.weight)

        embed = self.mlp(hs[0])

        x = self.conv(x)

        mask = torch.einsum('bnc,bchw->bnhw', embed, x0)
        # breakpoint()

        return mask, hs[0]

class MaskInsEncoder(nn.Module): # 0218
    def __init__(self, embed_channel=32, query_ins_num=48, depth_decoder=2, depth_encoder=0, out_seq_len=20, motion_category_num=2, classification_num=5):
        super(MaskInsEncoder, self).__init__()

        self.d_model = embed_channel

        self.query_num = query_ins_num
        self.out_seq_len = out_seq_len

        self.query_embed = torch.nn.Embedding(query_ins_num, embed_channel)

        # self.decoder = Transformer(d_model=embed_channel, nhead=8, num_encoder_layers=2,
        #          num_decoder_layers=2, dim_feedforward=64, dropout=0.1,
        #          activation="relu", normalize_before=False,
        #          return_intermediate_dec=False)

        self.depth_encoder = depth_encoder
        if depth_encoder > 0:
            encoder_layer = TransformerEncoderLayer(d_model=embed_channel, nhead=8, dim_feedforward=128,
                                                dropout=0.1, activation="relu", normalize_before=False)
            self.encoder = TransformerEncoder(encoder_layer, depth_encoder, None)

        decoder_layer = TransformerDecoderLayer(d_model=embed_channel, nhead=8, dim_feedforward=128,
                                                dropout=0.1, activation="relu", normalize_before=False)
        decoder_norm = nn.LayerNorm(embed_channel)
        self.decoder = TransformerDecoder(decoder_layer, depth_decoder, decoder_norm,
                                          return_intermediate=False)

        self.classification_mlp = nn.Sequential(
            nn.Linear(embed_channel, embed_channel),
            nn.ReLU(inplace=True),
            nn.Linear(embed_channel, classification_num)
        )

        self.motion_mlp = nn.Sequential(
            nn.Linear(embed_channel, embed_channel),
            nn.ReLU(inplace=True),
            nn.Linear(embed_channel, motion_category_num)
        )

        self.disp_mlp = nn.Sequential(
            nn.Linear(embed_channel, embed_channel),
            nn.ReLU(inplace=True),
            nn.Linear(embed_channel, out_seq_len * 2)
        )

        self.embedding_mlp = nn.Sequential(
            nn.Linear(embed_channel, embed_channel),
            nn.ReLU(inplace=True),
            nn.Linear(embed_channel, embed_channel)
        )

        self.pe_sine = PositionEncodingSinev2(embed_channel // 2)

        self.conv = nn.Sequential(
            nn.Conv2d(embed_channel, embed_channel, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_channel, embed_channel, 1, 1, 0)
        )

    def forward(self, x, x0):
        B = x.size(0)

        pos = self.pe_sine(x).flatten(2).permute(2, 0 ,1)

        tgt = torch.zeros((B, 1, self.d_model)).to(x.device).repeat(1, self.query_num, 1).permute(1, 0, 2)

        memory = x.flatten(2).permute(2, 0, 1)

        if self.depth_encoder > 0:
            memory = self.encoder(memory, pos=pos)

        hs = self.decoder(tgt, memory, pos=pos, query_pos=self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)).transpose(1, 2)

        ins_class_prob = self.classification_mlp(hs[0]).softmax(dim=-1) # B x N x class_num, class probability predictions, query-class
        ins_state_prob = self.motion_mlp(hs[0]).softmax(dim=-1) # B x N x motion_num, motion probability predictions, query-motion
        ins_disp_prob = self.disp_mlp(hs[0]).view(B, -1, self.out_seq_len, 2) # B x N x L x 2, motion displacement predictions, query-disp

        embed = self.embedding_mlp(hs[0]) # B x N x C, mask embeddings

        x0 = self.conv(x0)

        mask = torch.einsum('bnc,bchw->bnhw', embed, x0).sigmoid() # B x N x H x W, binary mask prediction, query-mask

        return ins_class_prob, ins_state_prob, ins_disp_prob, mask


class GridInsMotionNetFull(nn.Module):
    """
    raw_type: use instance-level features to enhanced the grid-level output
    """
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, ins_query_num=95):
        super(GridInsMotionNetFull, self).__init__()
        self.out_seq_len = out_seq_len

        self.cell_classify = CellClassification(in_channel=32 + 95)
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len, in_channel=32 + 95)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num, in_channel=32 + 95)
        self.stpn = STPN(height_feat_size=height_feat_size) # channel=32
        # self.ins_encoder = InsEncoder(embed_channel=32, ins_query_num=ins_query_num, depth_decoder=2, depth_encoder=2)
        self.ins_encoder = InsEncoder(embed_channel=32, ins_query_num=ins_query_num, depth_decoder=2, depth_encoder=2)

        self.ins_classify = nn.Sequential(
            nn.Conv2d(ins_query_num, ins_query_num, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins_query_num, ins_query_num, 1, 1, 0)
        )

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # Backbone network
        x = self.stpn(bevs) # (Batch, c, h, w)
        # x_small = torch.nn.functional.interpolate(x, scale_factor=(0.25,0.25), mode='bilinear') # (Batch, c, h/4, w/4)
        ins_x, hs = self.ins_encoder(x) # (Batch, N, h, w)
        # print(ins_x)



        # ins_prob, mask = self.ins_encoder(x_small) # ins_prob: (Batch, N, K) mask: (Batch, N, h/4, w/4)
        # mask = torch.nn.functional.interpolate(mask, scale_factor=(4,4), mode='bilinear') # (Batch, N, h, w)
        # segments = torch.einsum('bqhw,bqn->bnhw', mask, ins_prob) # (Batch, K, H, W)

        # ins_map = self.ins_classify(ins_x).softmax(dim=1) # (Batch, N, h, w), normalized instance map
        ins_map = torch.sigmoid(ins_x) # (Batch, N, h, w), normalized instance map
        ins_nums = ins_x.size(1) # N

        # mask_embed = hs[0]
        # breakpoint()
        grid_ins_feat = torch.cat([x, ins_x], dim=1)
        cell_class_pred = self.cell_classify(grid_ins_feat)
        state_class_pred = self.state_classify(grid_ins_feat)
        disp = self.motion_pred(grid_ins_feat)

        disp_pred = disp.view(-1, 2, x.size(-2), x.size(-1))

        return disp_pred, cell_class_pred, state_class_pred, ins_map

class GridInsMotionNet(nn.Module):
    """
    raw_type: use instance-level features to enhanced the grid-level output
    """
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, ins_query_num=95):
        super(GridInsMotionNet, self).__init__()
        self.out_seq_len = out_seq_len

        self.cell_classify = CellClassification(in_channel=32 + 95)
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len, in_channel=32 + 95)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num, in_channel=32 + 95)
        self.stpn = STPN(height_feat_size=height_feat_size) # channel=32
        # self.ins_encoder = InsEncoder(embed_channel=32, ins_query_num=ins_query_num, depth_decoder=2, depth_encoder=2)
        self.ins_encoder = InsEncoderV2(embed_channel=32, ins_query_num=ins_query_num, depth_decoder=2, depth_encoder=2)

        self.ins_classify = nn.Sequential(
            nn.Conv2d(ins_query_num, ins_query_num, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins_query_num, ins_query_num, 1, 1, 0)
        )

    def forward(self, bevs, return_bevs=False):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # Backbone network
        x = self.stpn(bevs) # (Batch, c, h, w)
        x_small = torch.nn.functional.interpolate(x, scale_factor=(0.25,0.25), mode='bilinear') # (Batch, c, h/4, w/4)
        ins_x, hs = self.ins_encoder(x_small, x) # (Batch, N, h, w)
        # print(ins_x)
        # ins_x = torch.nn.functional.interpolate(ins_x, scale_factor=(4,4), mode='bilinear') # (Batch, N, h, w)


        # ins_prob, mask = self.ins_encoder(x_small) # ins_prob: (Batch, N, K) mask: (Batch, N, h/4, w/4)
        # mask = torch.nn.functional.interpolate(mask, scale_factor=(4,4), mode='bilinear') # (Batch, N, h, w)
        # segments = torch.einsum('bqhw,bqn->bnhw', mask, ins_prob) # (Batch, K, H, W)

        # ins_map = self.ins_classify(ins_x).softmax(dim=1) # (Batch, N, h, w), normalized instance map
        ins_map = torch.sigmoid(ins_x) # (Batch, N, h, w), normalized instance map
        ins_nums = ins_x.size(1) # N

        # mask_embed = hs[0]
        # breakpoint()
        grid_ins_feat = torch.cat([x, ins_x], dim=1)
        cell_class_pred = self.cell_classify(grid_ins_feat)
        state_class_pred = self.state_classify(grid_ins_feat)
        disp = self.motion_pred(grid_ins_feat)

        disp_pred = disp.view(-1, 2, x.size(-2), x.size(-1))
        if return_bevs:
            return disp_pred, cell_class_pred, state_class_pred, ins_map, x
        return disp_pred, cell_class_pred, state_class_pred, ins_map

class InsMotionNet(nn.Module):
    """
    raw type: weighted sum to get disp pred, state pred, class pred in dense map
    """
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, ins_query_num=100):
        super(InsMotionNet, self).__init__()
        self.out_seq_len = out_seq_len

        self.cell_classify = CellClassification(in_channel=1)
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len, in_channel=1)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num, in_channel=1)
        self.stpn = STPN(height_feat_size=height_feat_size) # channel=32
        self.ins_encoder = InsEncoder(embed_channel=32, ins_query_num=ins_query_num, depth_decoder=2, depth_encoder=2)
        self.ins_classify = nn.Sequential(
            nn.Conv2d(ins_query_num, ins_query_num, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins_query_num, ins_query_num, 1, 1, 0)
        )

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # Backbone network
        x = self.stpn(bevs) # (Batch, c, h, w)
        x_small = torch.nn.functional.interpolate(x, scale_factor=(0.25,0.25), mode='bilinear') # (Batch, c, h/4, w/4)
        ins_x, hs = self.ins_encoder(x_small) # (Batch, N, h/4, w/4)
        # print(ins_x)
        ins_x = torch.nn.functional.interpolate(ins_x, scale_factor=(4,4), mode='bilinear') # (Batch, N, h, w)

        ins_map = self.ins_classify(ins_x).softmax(dim=1) # (Batch, N, h, w), normalized instance map

        ins_nums = ins_x.size(1) # N

        # Cell Classification head
        cell_class_pred = []
        for i in range(ins_nums):
            cell_class_pred.append(self.cell_classify(ins_x[:, i:i+1, :, :])) # (Batch, class_nums, h, w), cell classification for each instance feature map
        cell_class_pred = torch.stack(cell_class_pred, dim=1) # (Batch, N, class_nums, h, w)
        cell_class_pred = torch.einsum("bihw,bichw->bchw", ins_map, cell_class_pred) # (Batch, class_nums, h, w), weighted by normalized instance map

        # Motion State Classification head
        state_class_pred = []
        for i in range(ins_nums):
            state_class_pred.append(self.state_classify(ins_x[:, i:i+1, :, :])) # (Batch, motion_category_num, h, w), state classification for each instance feature map
        state_class_pred = torch.stack(state_class_pred, dim=1) # (Batch, N, motion_category_num, h, w)
        state_class_pred = torch.einsum("bihw,bichw->bchw", ins_map, state_class_pred) # (Batch, motion_category_num, h, w), weighted by normalized instance map

        # Motion Displacement prediction
        disp = []
        for i in range(ins_nums):
            disp.append(self.motion_pred(ins_x[:, i:i+1, :, :])) # (Batch, seq_len x 2, h, w), motion prediction for each instance feature map
        disp = torch.stack(disp, dim=1) # (Batch, N, seq_len x 2, h, w)
        disp = torch.einsum("bihw,bichw->bchw", ins_map, disp).contiguous() # (Batch, seq_len x 2, h, w), weighted by normalized instance map
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))

        return disp, cell_class_pred, state_class_pred, ins_map


class InsMotionNetv2_raw(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, ins_query_num=100):
        super(InsMotionNetv2_raw, self).__init__()
        self.out_seq_len = out_seq_len

        # self.cell_classify = CellClassification(in_channel=1)
        # self.motion_pred = MotionPrediction(seq_len=self.out_seq_len, in_channel=1)
        # self.state_classify = StateEstimation(motion_category_num=motion_category_num, in_channel=1)
        self.stpn = STPN(height_feat_size=height_feat_size) # channel=32
        self.ins_encoder = InsEncoder(embed_channel=32, ins_query_num=ins_query_num, depth_decoder=2, depth_encoder=2)
        self.ins_classify = nn.Sequential(
            nn.Conv2d(ins_query_num, ins_query_num, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins_query_num, ins_query_num, 1, 1, 0)
        )

        self.cell_classify = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 5)
        )

        self.motion_pred = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, self.out_seq_len * 2)
        )

        self.state_classify = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2)
        )

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # Backbone network
        x = self.stpn(bevs) # (Batch, c, h, w)
        x_small = torch.nn.functional.interpolate(x, scale_factor=(0.25,0.25), mode='bilinear') # (Batch, c, h/4, w/4)
        ins_x, ins_embeds = self.ins_encoder(x_small) # (Batch, N, h/4, w/4)
        ins_x = torch.nn.functional.interpolate(ins_x, scale_factor=(4,4), mode='bilinear') # (Batch, N, h, w)

        # ins_map = self.ins_classify(ins_x).softmax(dim=1) # (Batch, N, h, w), normalized instance map TODO: 这里要不要过ins_classify不太确定
        ins_map = torch.sigmoid(ins_x) # (Batch, N, h, w), normalized instance map
        b = ins_x.size(0)
        ins_nums = ins_x.size(1) # N

        cell_class_pred = self.cell_classify(ins_embeds) # (Batch, N, class_nums)
        state_pred = self.state_classify(ins_embeds) # (Batch, N, 2)
        disp_ = self.motion_pred(ins_embeds) # (Batch, N, seq_len * 2)
        disp = disp_.view(b, ins_nums, self.out_seq_len, 2) # (Batch, N, seq_len, 2)

        # post-process
        ins_map_idx = torch.argmax(ins_map, dim=1, keepdims=True)

        B, _, H, W = ins_map.shape
        cell_class_pred_map = torch.zeros((B, 5, H, W), device=ins_map.device)
        state_pred_map = torch.zeros((B, 2, H, W), device=ins_map.device)
        disp_map = torch.zeros((B, self.out_seq_len*2, H, W), device=ins_map.device)

        for b in range(B):
            for n in range(ins_nums):
                mask = (ins_map_idx[b, 0] == n) # (H, W)
                if torch.sum(mask) < 1:
                    continue

                cell_class_pred_map[b, :, mask] = cell_class_pred[b, n, :].unsqueeze(-1)
                state_pred_map[b, :, mask] = state_pred[b, n, :].unsqueeze(-1)
                disp_map[b, :, mask] = disp_[b, n, :].unsqueeze(-1)

        return ins_map, cell_class_pred, disp, state_pred, cell_class_pred_map, state_pred_map, disp_map.view(B, self.out_seq_len, 2, H, W)

class InsMotionNetv2(nn.Module):
    """
    Instance-level output
    """
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, ins_query_num=100):
        super(InsMotionNetv2, self).__init__()
        self.out_seq_len = out_seq_len

        # self.cell_classify = CellClassification(in_channel=1)
        # self.motion_pred = MotionPrediction(seq_len=self.out_seq_len, in_channel=1)
        # self.state_classify = StateEstimation(motion_category_num=motion_category_num, in_channel=1)
        self.stpn = STPN(height_feat_size=height_feat_size) # channel=32
        self.ins_encoder = InsEncoder(embed_channel=32, ins_query_num=ins_query_num, depth_decoder=2, depth_encoder=2)
        self.ins_classify = nn.Sequential(
            nn.Conv2d(ins_query_num, ins_query_num, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ins_query_num, ins_query_num, 1, 1, 0)
        )

        self.cell_classify = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 5)
        )

        self.motion_pred = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, self.out_seq_len * 2)
        )

        self.state_classify = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2)
        )

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # Backbone network
        x = self.stpn(bevs) # (Batch, c, h, w)
        x_small = torch.nn.functional.interpolate(x, scale_factor=(0.25,0.25), mode='bilinear') # (Batch, c, h/4, w/4)
        ins_x, ins_embeds = self.ins_encoder(x_small) # (Batch, N, h/4, w/4)
        ins_x = torch.nn.functional.interpolate(ins_x, scale_factor=(4,4), mode='bilinear') # (Batch, N, h, w)

        # ins_map = self.ins_classify(ins_x).softmax(dim=1) # (Batch, N, h, w), normalized instance map TODO: 这里要不要过ins_classify不太确定
        ins_map = torch.sigmoid(ins_x) # (Batch, N, h, w), normalized instance map
        b = ins_x.size(0)
        ins_nums = ins_x.size(1) # N

        cell_class_pred = self.cell_classify(ins_embeds) # (Batch, N, class_nums)
        state_pred = self.state_classify(ins_embeds) # (Batch, N, 2)
        disp_ = self.motion_pred(ins_embeds) # (Batch, N, seq_len * 2)
        disp = disp_.view(b, ins_nums, self.out_seq_len, 2) # (Batch, N, seq_len, 2)

        return ins_map, cell_class_pred, disp, state_pred

class LegoMotion(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, \
                ins_query_num=100):
        super(LegoMotion, self).__init__()
        self.out_seq_len = out_seq_len
        self.stpn = STPN(height_feat_size=height_feat_size) # channel=32
        self.mask_ed = MaskInsEncoder(embed_channel=32, query_ins_num=ins_query_num, depth_decoder=2, depth_encoder=0, \
                                      out_seq_len=out_seq_len, motion_category_num=motion_category_num, \
                                        classification_num=5)

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # Backbone network
        x = self.stpn(bevs) # (Batch, c, h, w)
        x_small = torch.nn.functional.interpolate(x, scale_factor=(0.25,0.25), mode='bilinear') # (Batch, c, h/4, w/4)
        ins_class_prob, ins_state_prob, ins_disp_prob, mask = self.mask_ed(x_small, x)

        return ins_class_prob, ins_state_prob, ins_disp_prob, mask

class LegoMotion_lowresX(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, \
                ins_query_num=100):
        super(LegoMotion_lowresX, self).__init__()
        self.out_seq_len = out_seq_len
        self.stpn = STPNv2(height_feat_size=height_feat_size) # channel=32
        self.encoder = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, 1, 1, 0)
        )
        self.mask_ed = MaskInsEncoder(embed_channel=32, query_ins_num=ins_query_num, depth_decoder=2, depth_encoder=0, \
                                      out_seq_len=out_seq_len, motion_category_num=motion_category_num, \
                                        classification_num=5)

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # Backbone network
        x_small, x = self.stpn(bevs) # x_small: (Batch, 512, h/8, w/8) x: (Batch, 32, h, w)
        x_small = self.encoder(x_small) # x_small: (Batch, 32, h/8, w/8)
        ins_class_prob, ins_state_prob, ins_disp_prob, mask = self.mask_ed(x_small, x)

        return ins_class_prob, ins_state_prob, ins_disp_prob, mask


class GeneralMotionNet(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, \
                ):
        super(GeneralMotionNet, self).__init__()
        self.out_seq_len = out_seq_len

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)
        self.stpn = STPN(height_feat_size=height_feat_size)

    def forward(self, bevs, prior_velocity=None, prior_direction=None):

        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        if prior_velocity is not None:
            prior_velocity = prior_velocity.permute(0, 1, 4, 2, 3)
            bevs = torch.cat([bevs, prior_velocity], dim=2)
        if prior_direction is not None:
            prior_direction = prior_direction.permute(0, 1, 4, 2, 3)
            bevs = torch.cat([bevs, prior_direction], dim=2)
        # Backbone network
        x = self.stpn(bevs)

        # Cell Classification head
        cell_class_pred = self.cell_classify(x)

        # Motion State Classification head
        state_class_pred = self.state_classify(x)

        # Motion Displacement prediction
        disp = self.motion_pred(x)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))

        return disp, cell_class_pred, state_class_pred

class LowerGeneralMotionNet(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, \
                ):
        super(LowerGeneralMotionNet, self).__init__()
        self.out_seq_len = out_seq_len

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)
        self.lowerstpn = LOWERSTPN(height_feat_size=height_feat_size)

    def forward(self, bevs, prior_velocity=None, prior_direction=None):

        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        if prior_velocity is not None:
            prior_velocity = prior_velocity.permute(0, 1, 4, 2, 3)
            bevs = torch.cat([bevs, prior_velocity], dim=2)
        if prior_direction is not None:
            prior_direction = prior_direction.permute(0, 1, 4, 2, 3)
            bevs = torch.cat([bevs, prior_direction], dim=2)
        # Backbone network
        x = self.lowerstpn(bevs)

        # Cell Classification head
        cell_class_pred = self.cell_classify(x)

        # Motion State Classification head
        state_class_pred = self.state_classify(x)

        # Motion Displacement prediction
        disp = self.motion_pred(x)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))

        return disp, cell_class_pred, state_class_pred

class MixGeneralMotionNet(nn.Module):

    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, \
                with_direction=False, with_cos=False):
        super(MixGeneralMotionNet, self).__init__()
        self.out_seq_len = out_seq_len

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)
        self.stpn = STPN(height_feat_size=height_feat_size)

        self.vencoder = VelocityEncoder(velocity_feat_size=1, with_direction=with_direction, with_cos = with_cos)
        # self.vencoder = VelocityEncoderV2(velocity_feat_size=1, with_direction=with_direction, with_cos=with_cos)
        self.ConvFuser = ConvFuser(in_channels=[32,32], out_channels=32)

        # SElayer
        self.se = SElayer(32)
    def forward(self, bevs, prior_velocity, prior_direction=None):

        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # if prior_velocity is not None:
        #     prior_velocity = prior_velocity.permute(0, 1, 4, 2, 3)
        #     bevs = torch.cat([bevs, prior_velocity], dim=2)
        if prior_direction is not None:
            # prior_direction = prior_direction.permute(0, 1, 4, 2, 3)
            prior_velocity = torch.cat([prior_velocity, prior_direction], dim=-1)
        # Backbone network
        bev_feat = self.stpn(bevs)
        v_feat = self.vencoder(prior_velocity)

        # Fuse
        # fuse_feat = torch.cat([bev_feat, v_feat],dim=1)
        fuse_feat = [bev_feat, v_feat]
        fuse_feat = self.ConvFuser(fuse_feat)
        # x = bev_feat + fuse_feat
        x = self.se(fuse_feat)
        # x = fuse_feat


        # Cell Classification head
        cell_class_pred = self.cell_classify(x)

        # Motion State Classification head
        state_class_pred = self.state_classify(x)

        # Motion Displacement prediction
        disp = self.motion_pred(x)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))

        return disp, cell_class_pred, state_class_pred

class VAE_GeneralMotionNet(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, \
                ):
        super(VAE_GeneralMotionNet, self).__init__()
        self.out_seq_len = out_seq_len

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)
        self.stpn = STPN(height_feat_size=height_feat_size)
        self.present_distrubution = SimpleVAE(in_channels=32, latent_dim=32, hidden_dims=[32, 64, 128, 256, 512],
                                    bev_size=256)

        self.future_distrubution = SimpleVAE(in_channels=32+5+2*out_seq_len+2, latent_dim=32, hidden_dims=[32, 64, 128, 256, 512],
                          bev_size=256)
        self.conv_fuser = ConvFuser(in_channels=[32, 32], out_channels=32)

    def forward(self, bevs, gt_map=None, prior_velocity=None, prior_direction=None):

        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        if prior_velocity is not None:
            prior_velocity = prior_velocity.permute(0, 1, 4, 2, 3)
            bevs = torch.cat([bevs, prior_velocity], dim=2)
        if prior_direction is not None:
            prior_direction = prior_direction.permute(0, 1, 4, 2, 3)
            bevs = torch.cat([bevs, prior_direction], dim=2)
        # Backbone network
        x = self.stpn(bevs)

        bev_refine, _, present_mu, present_log_sigma, _ = self.present_distrubution(x)

        # x_detach = x.detach()
        bev_gt_refine, vae_dict = None, None
        if gt_map is not None:

            con_x = torch.cat([x, gt_map], dim=1)
            bev_gt_refine, _, future_mu, future_log_sigma, _ = self.future_distrubution(con_x)

            vae_dict = {
                'present_mu': present_mu,
                'present_log_sigma': present_log_sigma,
                'future_mu': future_mu,
                'future_log_sigma': future_log_sigma,
            }

            # bev_gt_refine = self.future_distrubution(con_x)[0]
        if bev_gt_refine is not None:
            bev_refine = bev_gt_refine
        bev_refine = bev_refine + x
        # bev_refine = self.conv_fuser([x, bev_refine])
        # TODO: res connection
        """bev_refine=x + bev_refine"""

        # Cell Classification head
        cell_class_pred = self.cell_classify(bev_refine)

        # Motion State Classification head
        state_class_pred = self.state_classify(bev_refine)

        # Motion Displacement prediction
        disp = self.motion_pred(bev_refine)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))

        return disp, cell_class_pred, state_class_pred, vae_dict

# For MGDA loss computation
class FeatEncoder(nn.Module):
    def __init__(self, height_feat_size=13):
        super(FeatEncoder, self).__init__()
        self.stpn = STPN(height_feat_size=height_feat_size)

    def forward(self, bevs):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)
        x = self.stpn(bevs)

        return x


class MotionNetMGDA(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2):
        super(MotionNetMGDA, self).__init__()
        self.out_seq_len = out_seq_len

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)

    def forward(self, stpn_out):
        # Cell Classification head
        cell_class_pred = self.cell_classify(stpn_out)

        # Motion State Classification head
        state_class_pred = self.state_classify(stpn_out)

        # Motion Displacement prediction
        disp = self.motion_pred(stpn_out)
        disp = disp.view(-1, 2, stpn_out.size(-2), stpn_out.size(-1))

        return disp, cell_class_pred, state_class_pred


if __name__ == "__main__":
    # vecoder = VelocityEncoder(velocity_feat_size=1, with_direction=False, with_cos=False)
    # v = torch.rand((16, 5, 256, 256, 1))
    # print(vecoder(x=v).shape)

    # MaskEncoder
    v = torch.rand((16, 5, 256, 256, 13))
    model = LegoMotion()
    output = model(v)
    breakpoint()
    print(output.shape)
