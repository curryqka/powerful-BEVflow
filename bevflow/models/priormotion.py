"""PriorMotion（DSpG 部分）。从旧代码 powerfulmotion/model/ 逐行搬运。

来源：model/priormotion.py 的 PriorMotionSTPN（:269-344）、MotionPredictionV2、FuturePrediction、
ConvFuser、STPN(3 返回值)；model/vae_model.py 的 SimpleVAEV2；model/layers.py 的
ConvBlock/Bottleneck/SpatialGRU。属性名与旧代码一致（state_dict 兼容）。

与论文的差异（以代码为准）：论文 Fig 2 的 Dynamic/Static 双 VAE 在代码里是
present（观测先验，推理用）/future（未来后验，训练用）两个 SimpleVAEV2 组成的条件 VAE；
RVpE 不在本模型内（P6 第二步用 prior_extractor.py 的积木接线）。
"""
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from bevflow.models.motionnet import CellClassification, Conv3D, StateEstimation

HEIGHT_FEAT = 13


class MotionPredictionV2(nn.Module):
    def __init__(self, seq_len) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1_5 = nn.BatchNorm2d(32)

    def forward(self, x):
        b, seq, c, h, w = x.size()
        x = x.view(b * seq, c, h, w)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=(2, 2))
        x = F.relu(self.bn1_5(self.conv1_5(x)))
        x = F.interpolate(x, scale_factor=(2, 2))
        x = self.conv2(x)
        return x


class FuturePrediction(nn.Module):
    """SGRU 自回归特征生成：输入 (b, n_future, c, h, w) 与隐态 (b, c, h, w)。"""

    def __init__(self, in_channels, latent_dim, n_gru_blocks=3, n_res_layers=3):
        super().__init__()
        self.n_gru_blocks = n_gru_blocks
        self.spatial_grus = []
        self.res_blocks = []
        for i in range(self.n_gru_blocks):
            gru_in_channels = latent_dim if i == 0 else in_channels
            self.spatial_grus.append(SpatialGRU(gru_in_channels, in_channels))
            self.res_blocks.append(nn.Sequential(*[Bottleneck(in_channels) for _ in range(n_res_layers)]))
        self.spatial_grus = nn.ModuleList(self.spatial_grus)
        self.res_blocks = nn.ModuleList(self.res_blocks)

    def forward(self, x, hidden_state):
        for i in range(self.n_gru_blocks):
            x = self.spatial_grus[i](x, hidden_state, flow=None)
            b, n_future, c, h, w = x.shape
            x = self.res_blocks[i](x.view(b * n_future, c, h, w))
            x = x.view(b, n_future, c, h, w)
        return x


class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: List[int], out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


class ConvBlock(nn.Module):
    """2D 卷积 + 可选 BN/IN + 可选激活。"""

    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1,
                 bias=False, norm='bn', activation='relu', transpose=False):
        super().__init__()
        self.conv = nn.Conv2d if not transpose else __import__('functools').partial(
            nn.ConvTranspose2d, output_padding=1)
        self.conv = self.conv(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError('Invalid norm {}'.format(norm))
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh(inplace=True)
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError('Invalid activation {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class Bottleneck(nn.Module):
    """残差瓶颈块（旧代码 layers.py 原样）。"""

    def __init__(self, in_channels, out_channels=None, kernel_size=3, dilation=1, groups=1,
                 upsample=False, downsample=False, dropout=0.0):
        super().__init__()
        self._downsample = downsample
        bottleneck_channels = int(in_channels / 2)
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        assert dilation == 1
        if upsample:
            assert not downsample, 'downsample and upsample not possible simultaneously.'
            bottleneck_conv = nn.ConvTranspose2d(
                bottleneck_channels, bottleneck_channels, kernel_size=kernel_size,
                bias=False, dilation=1, stride=2, output_padding=padding_size,
                padding=padding_size, groups=groups)
        elif downsample:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels, bottleneck_channels, kernel_size=kernel_size,
                bias=False, dilation=dilation, stride=2, padding=padding_size, groups=groups)
        else:
            bottleneck_conv = nn.Conv2d(
                bottleneck_channels, bottleneck_channels, kernel_size=kernel_size,
                bias=False, dilation=dilation, padding=padding_size, groups=groups)

        self.layers = nn.Sequential(OrderedDict([
            ('conv_down_project', nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)),
            ('abn_down_project', nn.Sequential(nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True))),
            ('conv', bottleneck_conv),
            ('abn', nn.Sequential(nn.BatchNorm2d(bottleneck_channels), nn.ReLU(inplace=True))),
            ('conv_up_project', nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)),
            ('abn_up_project', nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))),
            ('dropout', nn.Dropout2d(p=dropout)),
        ]))

        if out_channels == in_channels and not downsample and not upsample:
            self.projection = None
        else:
            projection = OrderedDict()
            if upsample:
                projection.update({'upsample_skip_proj': Interpolate(scale_factor=2)})
            elif downsample:
                projection.update({'upsample_skip_proj': nn.MaxPool2d(kernel_size=2, stride=2)})
            projection.update({
                'conv_skip_proj': nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                'bn_skip_proj': nn.BatchNorm2d(out_channels),
            })
            self.projection = nn.Sequential(projection)

    def forward(self, *args):
        (x,) = args
        x_residual = self.layers(x)
        if self.projection is not None:
            if self._downsample:
                x = nn.functional.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
            return x_residual + self.projection(x)
        return x_residual + x


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class SpatialGRU(nn.Module):
    """卷积 GRU，输入 [B×T×C×H×W]，可选隐态。"""

    def __init__(self, input_size, hidden_size, gru_bias_init=0.0, norm='bn', activation='relu'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init
        self.conv_update = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde = ConvBlock(
            input_size + hidden_size, hidden_size, kernel_size=3, bias=False, norm=norm, activation=activation)

    def forward(self, x, state=None, flow=None, mode='bilinear'):
        assert len(x.size()) == 5, 'Input tensor must be BxTxCxHxW.'
        b, timesteps, c, h, w = x.size()
        assert c == self.input_size, f'feature sizes must match, got input {c}'
        rnn_output = []
        rnn_state = torch.zeros(b, self.hidden_size, h, w, device=x.device) if state is None else state
        for t in range(timesteps):
            x_t = x[:, t]
            rnn_state = self.gru_cell(x_t, rnn_state)
            rnn_output.append(rnn_state)
        return torch.stack(rnn_output, dim=1)

    def gru_cell(self, x, state):
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = torch.sigmoid(self.conv_update(x_and_state) + self.gru_bias_init)
        reset_gate = torch.sigmoid(self.conv_reset(x_and_state) + self.gru_bias_init)
        state_tilde = self.conv_state_tilde(torch.cat([x, (1.0 - reset_gate) * state], dim=1))
        return (1.0 - update_gate) * state + update_gate * state_tilde


class SimpleVAEV2(nn.Module):
    """空间隐变量 VAE：编码器 4×stride2 下采样，1×1 conv 出 mu/log_var，无解码器。"""

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None,
                 min_log_sigma: float = -5., max_log_sigma: float = 5., **kwargs) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 256]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.conv_mu = nn.Conv2d(hidden_dims[-1], out_channels=latent_dim, kernel_size=1)
        self.conv_var = nn.Conv2d(hidden_dims[-1], out_channels=latent_dim, kernel_size=1)

    def encode(self, input: torch.Tensor):
        result = self.encoder(input)
        return [self.conv_mu(result), self.conv_var(result)]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs):
        mu, log_var = self.encode(input)
        log_var = torch.clamp(log_var, self.min_log_sigma, self.max_log_sigma)
        z = self.reparameterize(mu, log_var)
        return [z, input, mu, log_var, z]


class STPN(nn.Module):
    """与 motionnet.py 的 STPN 同构，forward 额外返回 (x_6, x_7) 供 SGRU 隐态与后续使用。"""

    def __init__(self, height_feat_size=HEIGHT_FEAT):
        super().__init__()
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

        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)).contiguous()
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))
        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)).contiguous()
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))

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

        return res_x, x_6, x_7


class PriorMotionSTPN(nn.Module):
    """PriorMotion 主模型（DSpG）。训练时传 disp_gt_map（scene_gt 拼接图）走未来后验，
    推理时 disp_gt_map=None 从观测先验采样。"""

    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=HEIGHT_FEAT,
                 use_checkpoint=True, fusion="repeat", sgru_hidden_init="x6", latent_levels=4):
        super().__init__()
        self.out_seq_len = out_seq_len
        self.use_checkpoint = use_checkpoint
        self.fusion = fusion
        self.sgru_hidden_init = sgru_hidden_init
        # latent_levels=4 → 隐变量 16×16；3 → 32×32（探索项）
        vae_hidden = [32, 64, 128, 256][:latent_levels]

        self.stpn = STPN(height_feat_size=height_feat_size)
        self.cell_classify = CellClassification()
        self.motion_pred = MotionPredictionV2(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)

        self.present_distrubution = SimpleVAEV2(in_channels=32, latent_dim=32,
                                                hidden_dims=vae_hidden, bev_size=256)
        self.future_distrubution = SimpleVAEV2(in_channels=32 + 5 + 2 * out_seq_len + 2, latent_dim=32,
                                               hidden_dims=vae_hidden, bev_size=256)
        self.conv_fuser = ConvFuser(in_channels=[32, 32], out_channels=32)
        self.future_prediction = FuturePrediction(in_channels=32 * 4, latent_dim=32 * 4)

    def forward(self, bevs, disp_gt_map=None, prior_velocity=None, prior_direction=None):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        if prior_velocity is not None:
            prior_velocity = prior_velocity.permute(0, 1, 4, 2, 3)
            bevs = torch.cat([bevs, prior_velocity], dim=2)
        if prior_direction is not None:
            prior_direction = prior_direction.permute(0, 1, 4, 2, 3)
            bevs = torch.cat([bevs, prior_direction], dim=2)

        x, x_tiny, _ = self.stpn(bevs)

        bev_refine, _, present_mu, present_log_sigma, _ = self.present_distrubution(x)

        bev_gt_refine, vae_dict = None, None
        if disp_gt_map is not None:
            con_x = torch.cat([x, disp_gt_map], dim=1)
            bev_gt_refine, _, future_mu, future_log_sigma, _ = self.future_distrubution(con_x)
            vae_dict = {
                'present_mu': present_mu,
                'present_log_sigma': present_log_sigma,
                'future_mu': future_mu,
                'future_log_sigma': future_log_sigma,
            }

        if bev_gt_refine is not None:
            bev_refine = bev_gt_refine

        _, latent_dim, gauss_h, gauss_w = bev_refine.shape
        sample = bev_refine.unsqueeze(1).repeat(1, self.out_seq_len, 4, 4, 4)
        hidden = x_tiny if self.sgru_hidden_init == "x6" else None
        if self.use_checkpoint and self.training:
            # SGRU T=20 展开的激活图远超 40GB 卡：检查点重算换显存，数值不变
            import torch.utils.checkpoint as cp

            disp_feat = cp.checkpoint(self.future_prediction, sample, hidden, use_reentrant=False)
        else:
            disp_feat = self.future_prediction(sample, hidden)

        if self.fusion == "conv_fuser":
            bev_refine = F.interpolate(bev_refine, size=(256, 256), mode="nearest")
            bev_refine = self.conv_fuser([x, bev_refine])
        else:
            bev_refine = bev_refine.repeat(1, 1, int(256 / gauss_h), int(256 / gauss_w)) + x

        cell_class_pred = self.cell_classify(bev_refine)
        state_class_pred = self.state_classify(bev_refine)
        disp = self.motion_pred(disp_feat)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))

        return disp, cell_class_pred, state_class_pred, vae_dict
