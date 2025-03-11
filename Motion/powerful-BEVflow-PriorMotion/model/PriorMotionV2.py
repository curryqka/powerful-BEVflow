# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import List
from model.attention import SElayer, CAlayer
from model.gan_model import NoiseGenerator
from model.vae_model import BetaVAE, SimpleVAE, SimpleVAEV2, DualBranchVAE, DualBranchVAEv2
from model.layers import Bottleneck,SpatialGRU
from model.prior_extractor import SimplePositionEncoding, SineContinuousPositionalEncoding, DispEncoding, GridInsTransformer, \
                                    MaskedCrossAttention, GridPriorv2
from model.be_sti import STPN as STI
from model.attention import GlobalLocalAttention
from model.model_aug import rotate_tensor, inverse_rotate_tensor
class CellClassification(nn.Module):
    def __init__(self, category_num=5):
        super(CellClassification, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, category_num, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class StateEstimation(nn.Module):
    def __init__(self, motion_category_num=2):
        super(StateEstimation, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, motion_category_num, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class MotionPrediction(nn.Module):
    def __init__(self, seq_len):
        super(MotionPrediction, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 2 * seq_len, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x

class MotionPredictionV2(nn.Module):
    def __init__(self, seq_len) -> None:
        super(MotionPredictionV2, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0)
        # self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 输出 (64,128,128)
        # self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 输出 (32,256,256)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1_5 = nn.BatchNorm2d(32)

    def forward(self, x):
        b, seq, c, h, w = x.size()
        x = x.view(b * seq, c, h, w)


        # x = self.deconv1(x)
        # x = self.deconv2(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=(2, 2))
        x = F.relu(self.bn1_5(self.conv1_5(x)))
        x = F.interpolate(x, scale_factor=(2, 2))
        x = self.conv2(x)
        # breakpoint()
        return x
class FuturePrediction(nn.Module):
    """future prediction with grus"""
    def __init__(self, in_channels, latent_dim, n_gru_blocks=3, n_res_layers=3):
        super().__init__()
        self.n_gru_blocks = n_gru_blocks

        self.spatial_grus = []
        self.res_blocks = []

        for i in range(self.n_gru_blocks):
            gru_in_channels = latent_dim if i == 0 else in_channels
            self.spatial_grus.append(SpatialGRU(gru_in_channels, in_channels))
            self.res_blocks.append(torch.nn.Sequential(*[Bottleneck(in_channels)
                                                         for _ in range(n_res_layers)]))

        self.spatial_grus = torch.nn.ModuleList(self.spatial_grus)
        self.res_blocks = torch.nn.ModuleList(self.res_blocks)

    def forward(self, x, hidden_state):
        # x has shape (b, n_future, c, h, w), hidden_state (b, c, h, w)
        for i in range(self.n_gru_blocks):
            x = self.spatial_grus[i](x, hidden_state, flow=None)
            b, n_future, c, h, w = x.shape

            x = self.res_blocks[i](x.view(b * n_future, c, h, w))
            x = x.view(b, n_future, c, h, w)

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

        return res_x, x_6, x_7


"""For VAE generator with spatialGRU with prior knowledge extractor with dual"""
class PriorMotionv2(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, num_grid=100):
        super(PriorMotionv2, self).__init__()
        self.out_seq_len = out_seq_len

        self.stpn = STPN(height_feat_size=height_feat_size)
        self.cell_classify = CellClassification()
        self.motion_pred = MotionPredictionV2(seq_len=out_seq_len)
        self.motion_pred_stat = MotionPrediction(seq_len=out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)

        """prior info grid  prior"""
        # self.grid_prior = GridPrior()
        self.grid_prior = GridPriorv2(seq_len=out_seq_len, cell_input_dim=16, num_cls=5, num_state=motion_category_num)
        """prior info extractor"""
        # grid positional encoding
        self.grid_pos_encoding = SineContinuousPositionalEncoding(
                                num_feats=16,  # 2 * 16 = 32 final dim
                                temperature=1000,
                                normalize=True,
                                range=[128, 128],
                                offset=[128, 128],
                            )
        # instance encoder
        self.instance_attn = GridInsTransformer(
                            input_dim=32*num_grid + 32*out_seq_len + 5,  # 32 * 11 + 8
                            dmodel=32,      # TODO: maybe change?
                            hidden_dim=256,  # TODO: maybe change?
                            nheads=4,
                            nlayers=6,
                            batch_first=True,
                            pos_encoder="sin",
                            disp_encoder="temporal",
                        )

        self.bev_pos_encoding = SimplePositionEncoding(num_feats=16, row_num_embed=256, col_num_embed=256)
        self.bev_attn = MaskedCrossAttention(embed_dims=32,
                                            num_heads=8,
                                            dropout=0.2,
                                            init_cfg=None,
                                            batch_first=True
                                                ) # add layernorm?

        self.present_distrubution = DualBranchVAEv2(in_channels=32, latent_dim=32, hidden_dims=[32, 64, 128, 256],
                                    bev_size=256)

        self.future_distrubution = DualBranchVAEv2(in_channels=32+40+ 2*out_seq_len + 5 + 2, latent_dim=32, hidden_dims=[32, 64, 128, 256],
                          bev_size=256)
        self.conv_fuser = ConvFuser(in_channels=[32, 32], out_channels=32)
        self.future_prediction = FuturePrediction(in_channels=32*4, latent_dim=32*4)
    def forward(self, bevs, disp_gt_map=None, instance_info=None, dir_aug=False, grid_encoder=False, non_empty_map=None, \
                motion_state=None, training_only=True):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # motion_gt_clone = disp_gt_map[-1]
        if motion_state.shape[1] != 2:
            motion_state = motion_state.permute(0, 3, 1, 2)
        motion_gt_clone = motion_state.to(bevs.device)
        motion_gt_state = torch.argmax(motion_gt_clone, dim=1) + 1
        if non_empty_map is not None:
            non_empty_map = non_empty_map.view(-1, bevs.shape[-2], bevs.shape[-1]).to(bevs.device)
            motion_gt_state = (non_empty_map * motion_gt_state).to(torch.int32)
        dyn_mask = (motion_gt_state==2)
        stat_mask = (motion_gt_state==1)

        motion_masks = (dyn_mask, stat_mask)
        # Backbone network
        x, x_tiny, _ = self.stpn(bevs)

        bs, bev_dim, bev_h, bev_w = x.size()

        bev_prior_feat = None

        """Instance information"""
        if instance_info is not None:
            # instance info extractor with self-attn
            (ins_pe_list, ins_disp_list, onehot_category_list) = instance_info
            break_flag = False
            for ins in ins_pe_list:
                if ins.shape[0] == 0 or ins.shape[0] == 1:
                    bev_prior_feat = None
                    break_flag = True
            if not break_flag:
                ins_feat = self.instance_attn(ins_pe_list, ins_disp_list, onehot_category_list, device=x.device)

                if isinstance(ins_feat, list):
                    # batch, num_polylines * feat_dim
                    ins_graph_shapes = torch.tensor([mfeat.shape[0] for mfeat in ins_feat],
                                                    device=ins_feat[0].device)

                    max_num_polylines = ins_graph_shapes.max()
                    # breakpoint()
                    ins_feat = torch.stack([torch.cat([mfeat, mfeat.new_zeros(max_num_polylines - mfeat.shape[0], mfeat.shape[1])], dim=0) for mfeat in ins_feat])  # batch, max_num_polylines, feat_dim
                bev_mask = torch.zeros((bs, bev_h, bev_w)).to(x.device)
                bev_pos = self.bev_pos_encoding(bev_mask)
                # breakpoint()
                bev_pos = bev_pos.flatten(2).permute(2, 0, 1).permute(1, 0, 2)

                bev = x.clone()
                bev = bev.view(-1, bev_dim, bev_w*bev_h).permute(0, 2, 1)
                bev_prior_feat = self.bev_attn(bev, ins_feat, ins_feat, query_pos=bev_pos, residual=bev, spatial_shapes=ins_graph_shapes)

        """present distribution
        {
            'dynamic': (z_dyn, mu_dyn, logvar_dyn),
            'static': (z_stat, mu_stat, logvar_stat),
            'input': input
        }"""
        # bev_refine, _, present_mu, present_log_sigma, _ = self.present_distrubution(x)
        present_distr_dict = self.present_distrubution(x, motion_masks)
        z_dyn = present_distr_dict['dynamic'][0]
        present_mu_dyn, present_logvar_dyn = present_distr_dict['dynamic'][1], present_distr_dict['dynamic'][2]

        # x_detach = x.detach()
        future_z_dynamic_distribution, vae_dict = None, None
        if disp_gt_map is not None:
            if instance_info is not None and bev_prior_feat is not None:
                bev_prior_feat = bev_prior_feat.contiguous().view(-1, bev_dim, bev_w, bev_h)
                # bev_prior_feat = bev_prior_feat.reshape(-1, bev_dim, bev_w, bev_h)
                # bev_prior_feat = bev_prior_feat + x
                if grid_encoder:

                    (pixel_cat_map_gt_clone, all_disp_field_gt_clone, motion_gt_clone) = disp_gt_map
                    all_disp_field_gt_clone = all_disp_field_gt_clone.view(bevs.shape[0], self.out_seq_len, bevs.shape[-2], bevs.shape[-1], -1)
                    gt_gird_feat = self.grid_prior(all_disp_field_gt_clone, pixel_cat_map_gt_clone, motion_gt_clone)
                    con_x = torch.cat([bev_prior_feat, gt_gird_feat, *disp_gt_map], dim=1)
                else:

                    con_x = torch.cat([bev_prior_feat, *disp_gt_map], dim=1)
            else:
                if grid_encoder:
                    (pixel_cat_map_gt_clone, all_disp_field_gt_clone, motion_gt_clone) = disp_gt_map
                    all_disp_field_gt_clone = all_disp_field_gt_clone.view(bevs.shape[0], self.out_seq_len, bevs.shape[-2], bevs.shape[-1], -1)
                    gt_gird_feat = self.grid_prior(all_disp_field_gt_clone, pixel_cat_map_gt_clone, motion_gt_clone)
                    con_x = torch.cat([x, gt_gird_feat, *disp_gt_map], dim=1)
                else:
                    con_x = torch.cat([x, *disp_gt_map], dim=1)
                # con_x = torch.cat([x, disp_gt_map], dim=1)
        else:
            # con_x = torch.cat()
            pass
        if training_only:
            """gt bev latent distribution"""
            # bev_gt_refine, _, future_mu, future_log_sigma, _ = self.future_distrubution(con_x)
            future_distr_dict = self.future_distrubution(con_x, motion_masks)
            future_z_dynamic_distribution = future_distr_dict['dynamic'][0]
            future_mu_dyn, future_logvar_dyn = future_distr_dict['dynamic'][1], future_distr_dict['dynamic'][2]

            vae_dict = {
                'present_mu_dyn': present_mu_dyn,
                'present_log_sigma_dyn': present_logvar_dyn,
                'future_mu_dyn': future_mu_dyn,
                'future_log_sigma_dyn': future_logvar_dyn,
            }

        if future_z_dynamic_distribution is not None:
            z_dyn = future_z_dynamic_distribution
        # bev_refine: [bs, latent, h, w]
        # bev_refine: [bs, latent, 1, 1]
        # breakpoint()
        # latent=32
        _, latent_dim, gauss_h, gauss_w = z_dyn.shape
        auto_z = z_dyn.unsqueeze(1).repeat(1, self.out_seq_len, 4, 4, 4)
        # breakpoint()
        disp_feat = self.future_prediction(auto_z, x_tiny)

        # bev_refine = bev_refine.expand(-1, x.size(-3), x.size(-2), x.size(-1)) + x
        """motion mask divided"""
        z_dyn = z_dyn.repeat(1, 1, int(256/gauss_h), int(256/gauss_w) ) + x


        # TODO: res connection
        """bev_refine=x + bev_refine"""
        # Cell Classification head
        cell_class_pred = self.cell_classify(z_dyn)

        # Motion State Classification head
        state_class_pred = self.state_classify(z_dyn)
        # Dual Motion Displacement prediction
        disp = self.motion_pred(disp_feat)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))


        disp_stat = self.motion_pred_stat(z_dyn)
        disp_stat = disp_stat.view(-1, 2, x.size(-2), x.size(-1))

        # data augmentation with direction
        # rotation disp list
        disp_rotated_list = []
        if dir_aug:
            # rotation angles
            angles = [90, 180, 270]
            disp_clone = disp.clone()
            disp_clone = disp.view(bs, -1, 2, x.size(-2), x.size(-1))
            T = disp_clone.shape[1]


            for angle in angles:
                # rotate bev
                rotated_bev_feature = rotate_tensor(disp_feat, angle)

                # decoder disp with rotation
                rotated_disp_pred = self.motion_pred(rotated_bev_feature)

                # rotated_disp_pred reshape
                rotated_disp_pred = rotated_disp_pred.view(bs, T, -1, x.size(-2), x.size(-1))

                # inverse rotation
                inv_rotated_disp_pred = inverse_rotate_tensor(rotated_disp_pred, angle)
                assert inv_rotated_disp_pred.shape == disp_clone.shape, f"Shapes do not match: {inv_rotated_disp_pred.shape} != {disp_gt.shape}"

                # inverse rotated disp pred shape
                inv_rotated_disp_pred = inv_rotated_disp_pred.view(-1, 2, x.size(-2), x.size(-1))
                disp_rotated_list.append(inv_rotated_disp_pred)

        return disp, disp_stat, cell_class_pred, state_class_pred, vae_dict, x

"""For VAE generator with spatialGRU with prior knowledge extractor with dual"""
class PriorMotionSTIv2(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, num_grid=100):
        super(PriorMotionSTIv2, self).__init__()
        self.out_seq_len = out_seq_len

        self.stpn = STI(height_feat_size=height_feat_size)
        self.cell_classify = CellClassification()
        self.motion_pred = MotionPredictionV2(seq_len=out_seq_len)
        self.motion_pred_stat = MotionPrediction(seq_len=out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)

        """prior info grid  prior"""
        # self.grid_prior = GridPrior()
        self.grid_prior = GridPriorv2(seq_len=out_seq_len, cell_input_dim=16, num_cls=5, num_state=motion_category_num)
        """prior info extractor"""
        # grid positional encoding
        self.grid_pos_encoding = SineContinuousPositionalEncoding(
                                num_feats=16,  # 2 * 16 = 32 final dim
                                temperature=1000,
                                normalize=True,
                                range=[128, 128],
                                offset=[128, 128],
                            )
        # instance encoder
        self.instance_attn = GridInsTransformer(
                            input_dim=32*num_grid + 32*out_seq_len + 5,  # 32 * 11 + 8
                            dmodel=32,      # TODO: maybe change?
                            hidden_dim=256,  # TODO: maybe change?
                            nheads=4,
                            nlayers=6,
                            batch_first=True,
                            pos_encoder="sin",
                            disp_encoder="temporal",
                        )

        self.bev_pos_encoding = SimplePositionEncoding(num_feats=16, row_num_embed=256, col_num_embed=256)
        self.bev_attn = MaskedCrossAttention(embed_dims=32,
                                            num_heads=8,
                                            dropout=0.2,
                                            init_cfg=None,
                                            batch_first=True
                                                ) # add layernorm?

        self.present_distrubution = DualBranchVAEv2(in_channels=32, latent_dim=32, hidden_dims=[32, 64, 128, 256],
                                    bev_size=256)

        self.future_distrubution = DualBranchVAEv2(in_channels=32+40+ 2*out_seq_len + 5 + 2, latent_dim=32, hidden_dims=[32, 64, 128, 256],
                          bev_size=256)
        self.present_gllayer = GlobalLocalAttention(channel=32)
        self.future_gllayer = GlobalLocalAttention(channel=32+40+ 2*out_seq_len + 5 + 2,)
        self.conv_fuser = ConvFuser(in_channels=[32, 32], out_channels=32)
        self.future_prediction = FuturePrediction(in_channels=32*4, latent_dim=32*4)
    def forward(self, bevs, disp_gt_map=None, instance_info=None, dir_aug=False, grid_encoder=False, non_empty_map=None, \
                motion_state=None, training_only=True):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # motion_gt_clone = disp_gt_map[-1]
        if motion_state.shape[1] != 2:
            motion_state = motion_state.permute(0, 3, 1, 2)
        motion_gt_clone = motion_state.to(bevs.device)
        motion_gt_state = torch.argmax(motion_gt_clone, dim=1) + 1
        if non_empty_map is not None:
            non_empty_map = non_empty_map.view(-1, bevs.shape[-2], bevs.shape[-1]).to(bevs.device)
            motion_gt_state = (non_empty_map * motion_gt_state).to(torch.int32)
        dyn_mask = (motion_gt_state==2)
        stat_mask = (motion_gt_state==1)

        motion_masks = (dyn_mask, stat_mask)
        # Backbone network
        x, x_tiny, _ = self.stpn(bevs)

        x = self.present_gllayer(x) + x

        bs, bev_dim, bev_h, bev_w = x.size()

        bev_prior_feat = None

        """Instance information"""
        if instance_info is not None:
            # instance info extractor with self-attn
            (ins_pe_list, ins_disp_list, onehot_category_list) = instance_info
            break_flag = False
            for ins in ins_pe_list:
                if ins.shape[0] == 0 or ins.shape[0] == 1:
                    bev_prior_feat = None
                    break_flag = True
            if not break_flag:
                ins_feat = self.instance_attn(ins_pe_list, ins_disp_list, onehot_category_list, device=x.device)

                if isinstance(ins_feat, list):
                    # batch, num_polylines * feat_dim
                    ins_graph_shapes = torch.tensor([mfeat.shape[0] for mfeat in ins_feat],
                                                    device=ins_feat[0].device)

                    max_num_polylines = ins_graph_shapes.max()
                    # breakpoint()
                    ins_feat = torch.stack([torch.cat([mfeat, mfeat.new_zeros(max_num_polylines - mfeat.shape[0], mfeat.shape[1])], dim=0) for mfeat in ins_feat])  # batch, max_num_polylines, feat_dim
                bev_mask = torch.zeros((bs, bev_h, bev_w)).to(x.device)
                bev_pos = self.bev_pos_encoding(bev_mask)
                # breakpoint()
                bev_pos = bev_pos.flatten(2).permute(2, 0, 1).permute(1, 0, 2)

                bev = x.clone()
                bev = bev.view(-1, bev_dim, bev_w*bev_h).permute(0, 2, 1)
                bev_prior_feat = self.bev_attn(bev, ins_feat, ins_feat, query_pos=bev_pos, residual=bev, spatial_shapes=ins_graph_shapes)

        """present distribution
        {
            'dynamic': (z_dyn, mu_dyn, logvar_dyn),
            'static': (z_stat, mu_stat, logvar_stat),
            'input': input
        }"""
        # bev_refine, _, present_mu, present_log_sigma, _ = self.present_distrubution(x)
        present_distr_dict = self.present_distrubution(x, motion_masks)
        z_dyn = present_distr_dict['dynamic'][0]
        present_mu_dyn, present_logvar_dyn = present_distr_dict['dynamic'][1], present_distr_dict['dynamic'][2]

        # x_detach = x.detach()
        future_z_dynamic_distribution, vae_dict, con_x = None, None, None
        if disp_gt_map is not None:
            if instance_info is not None and bev_prior_feat is not None:
                bev_prior_feat = bev_prior_feat.contiguous().view(-1, bev_dim, bev_w, bev_h)
                # bev_prior_feat = bev_prior_feat.reshape(-1, bev_dim, bev_w, bev_h)
                # bev_prior_feat = bev_prior_feat + x
                if grid_encoder:

                    (pixel_cat_map_gt_clone, all_disp_field_gt_clone, motion_gt_clone) = disp_gt_map
                    all_disp_field_gt_clone = all_disp_field_gt_clone.view(bevs.shape[0], self.out_seq_len, bevs.shape[-2], bevs.shape[-1], -1)
                    gt_gird_feat = self.grid_prior(all_disp_field_gt_clone, pixel_cat_map_gt_clone, motion_gt_clone)
                    con_x = torch.cat([bev_prior_feat, gt_gird_feat, *disp_gt_map], dim=1)
                    con_x = self.future_gllayer(con_x) + con_x
                else:

                    con_x = torch.cat([bev_prior_feat, *disp_gt_map], dim=1)
            else:
                if grid_encoder:
                    (pixel_cat_map_gt_clone, all_disp_field_gt_clone, motion_gt_clone) = disp_gt_map
                    all_disp_field_gt_clone = all_disp_field_gt_clone.view(bevs.shape[0], self.out_seq_len, bevs.shape[-2], bevs.shape[-1], -1)
                    gt_gird_feat = self.grid_prior(all_disp_field_gt_clone, pixel_cat_map_gt_clone, motion_gt_clone)
                    con_x = torch.cat([x, gt_gird_feat, *disp_gt_map], dim=1)
                    con_x = self.future_gllayer(con_x) + con_x
                else:
                    con_x = torch.cat([x, *disp_gt_map], dim=1)
                # con_x = torch.cat([x, disp_gt_map], dim=1)
        else:
            # con_x = torch.cat()
            pass

        if training_only:
            """gt bev latent distribution"""
            # bev_gt_refine, _, future_mu, future_log_sigma, _ = self.future_distrubution(con_x)
            future_distr_dict = self.future_distrubution(con_x, motion_masks)
            future_z_dynamic_distribution = future_distr_dict['dynamic'][0]
            future_mu_dyn, future_logvar_dyn = future_distr_dict['dynamic'][1], future_distr_dict['dynamic'][2]


            vae_dict = {
                'present_mu_dyn': present_mu_dyn,
                'present_log_sigma_dyn': present_logvar_dyn,
                'future_mu_dyn': future_mu_dyn,
                'future_log_sigma_dyn': future_logvar_dyn,
            }

        if future_z_dynamic_distribution is not None:
            z_dyn = future_z_dynamic_distribution

        # bev_refine: [bs, latent, h, w]
        # bev_refine: [bs, latent, 1, 1]
        # breakpoint()
        # latent=32
        _, latent_dim, gauss_h, gauss_w = z_dyn.shape
        auto_z = z_dyn.unsqueeze(1).repeat(1, self.out_seq_len, 4, 4, 4)
        # breakpoint()
        disp_feat = self.future_prediction(auto_z, x_tiny)

        # bev_refine = bev_refine.expand(-1, x.size(-3), x.size(-2), x.size(-1)) + x
        """motion mask divided"""
        z_dyn = z_dyn.repeat(1, 1, int(256/gauss_h), int(256/gauss_w) ) + x


        # TODO: res connection
        """bev_refine=x + bev_refine"""
        # Cell Classification head
        cell_class_pred = self.cell_classify(z_dyn)

        # Motion State Classification head
        state_class_pred = self.state_classify(z_dyn)
        # Dual Motion Displacement prediction
        disp = self.motion_pred(disp_feat)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))


        disp_stat = self.motion_pred_stat(z_dyn)
        disp_stat = disp_stat.view(-1, 2, x.size(-2), x.size(-1))

        # data augmentation with direction
        # rotation disp list
        disp_rotated_list = []
        if dir_aug:
            # rotation angles
            angles = [90, 180, 270]
            disp_clone = disp.clone()
            disp_clone = disp.view(bs, -1, 2, x.size(-2), x.size(-1))
            T = disp_clone.shape[1]


            for angle in angles:
                # rotate bev
                rotated_bev_feature = rotate_tensor(disp_feat, angle)

                # decoder disp with rotation
                rotated_disp_pred = self.motion_pred(rotated_bev_feature)

                # rotated_disp_pred reshape
                rotated_disp_pred = rotated_disp_pred.view(bs, T, -1, x.size(-2), x.size(-1))

                # inverse rotation
                inv_rotated_disp_pred = inverse_rotate_tensor(rotated_disp_pred, angle)
                assert inv_rotated_disp_pred.shape == disp_clone.shape, f"Shapes do not match: {inv_rotated_disp_pred.shape} != {disp_gt.shape}"

                # inverse rotated disp pred shape
                inv_rotated_disp_pred = inv_rotated_disp_pred.view(-1, 2, x.size(-2), x.size(-1))
                disp_rotated_list.append(inv_rotated_disp_pred)

        return disp, disp_stat, cell_class_pred, state_class_pred, vae_dict, x

"""For VAE generator with spatialGRU with prior knowledge extractor with dual"""
class PriorMotionInsv2(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, num_grid=100):
        super(PriorMotionInsv2, self).__init__()
        self.out_seq_len = out_seq_len

        self.stpn = STPN(height_feat_size=height_feat_size)
        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=out_seq_len)
        self.motion_pred_stat = MotionPrediction(seq_len=out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)

        """prior info extractor"""
        # grid positional encoding
        self.grid_pos_encoding = SineContinuousPositionalEncoding(
                                num_feats=16,  # 2 * 16 = 32 final dim
                                temperature=1000,
                                normalize=True,
                                range=[128, 128],
                                offset=[128, 128],
                            )
        # instance encoder
        self.instance_attn = GridInsTransformer(
                            input_dim=32*num_grid + 32*out_seq_len + 5,  # 32 * 11 + 8
                            dmodel=32,      # TODO: maybe change?
                            hidden_dim=256,  # TODO: maybe change?
                            nheads=4,
                            nlayers=6,
                            batch_first=True,
                            pos_encoder="sin",
                            disp_encoder="temporal",
                        )

        self.bev_pos_encoding = SimplePositionEncoding(num_feats=16, row_num_embed=256, col_num_embed=256)
        self.bev_attn = MaskedCrossAttention(embed_dims=32,
                                            num_heads=8,
                                            dropout=0.2,
                                            init_cfg=None,
                                            batch_first=True
                                                ) # add layernorm?

        self.present_distrubution = DualBranchVAEv2(in_channels=32, latent_dim=32, hidden_dims=[32, 64, 128, 256],
                                    bev_size=256)

        self.future_distrubution = DualBranchVAEv2(in_channels=32+32, latent_dim=32, hidden_dims=[32, 64, 128, 256],
                          bev_size=256)
        self.conv_fuser = ConvFuser(in_channels=[32, 32], out_channels=32)
        # self.future_prediction = FuturePrediction(in_channels=32*4, latent_dim=32*4)
    def forward(self, bevs, disp_gt_map=None, instance_info=None, dir_aug=False, grid_encoder=False, non_empty_map=None, motion_state=None):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # motion_gt_clone = disp_gt_map[-1]
        if motion_state.shape[1] != 2:
            motion_state = motion_state.permute(0, 3, 1, 2)
        motion_gt_clone = motion_state.to(bevs.device)
        motion_gt_state = torch.argmax(motion_gt_clone, dim=1) + 1
        if non_empty_map is not None:
            non_empty_map = non_empty_map.view(-1, bevs.shape[-2], bevs.shape[-1]).to(bevs.device)
            motion_gt_state = (non_empty_map * motion_gt_state).to(torch.int32)
        dyn_mask = (motion_gt_state==2)
        stat_mask = (motion_gt_state==1)

        motion_masks = (dyn_mask, stat_mask)
        # Backbone network
        x, x_tiny, _ = self.stpn(bevs)

        bs, bev_dim, bev_h, bev_w = x.size()

        bev_prior_feat = None

        """Instance information"""
        if instance_info is not None:
            # instance info extractor with self-attn
            (ins_pe_list, ins_disp_list, onehot_category_list) = instance_info
            break_flag = False
            for ins in ins_pe_list:
                if ins.shape[0] == 0 or ins.shape[0] == 1:
                    bev_prior_feat = None
                    break_flag = True
            if not break_flag:
                ins_feat = self.instance_attn(ins_pe_list, ins_disp_list, onehot_category_list, device=x.device)

                if isinstance(ins_feat, list):
                    # batch, num_polylines * feat_dim
                    ins_graph_shapes = torch.tensor([mfeat.shape[0] for mfeat in ins_feat],
                                                    device=ins_feat[0].device)

                    max_num_polylines = ins_graph_shapes.max()
                    # breakpoint()
                    ins_feat = torch.stack([torch.cat([mfeat, mfeat.new_zeros(max_num_polylines - mfeat.shape[0], mfeat.shape[1])], dim=0) for mfeat in ins_feat])  # batch, max_num_polylines, feat_dim
                bev_mask = torch.zeros((bs, bev_h, bev_w)).to(x.device)
                bev_pos = self.bev_pos_encoding(bev_mask)
                # breakpoint()
                bev_pos = bev_pos.flatten(2).permute(2, 0, 1).permute(1, 0, 2)

                bev = x.clone()
                bev = bev.view(-1, bev_dim, bev_w*bev_h).permute(0, 2, 1)
                bev_prior_feat = self.bev_attn(bev, ins_feat, ins_feat, query_pos=bev_pos, residual=bev, spatial_shapes=ins_graph_shapes)

        """present distribution
        {
            'dynamic': (z_dyn, mu_dyn, logvar_dyn),
            'static': (z_stat, mu_stat, logvar_stat),
            'input': input
        }"""
        # bev_refine, _, present_mu, present_log_sigma, _ = self.present_distrubution(x)
        present_distr_dict = self.present_distrubution(x, motion_masks)
        z_dyn = present_distr_dict['dynamic'][0]

        present_mu_dyn, present_logvar_dyn = present_distr_dict['dynamic'][1], present_distr_dict['dynamic'][2]


        # x_detach = x.detach()
        future_z_dynamic_distribution, vae_dict = None, None
        if disp_gt_map is not None:
            if instance_info is not None and bev_prior_feat is not None:
                bev_prior_feat = bev_prior_feat.contiguous().view(-1, bev_dim, bev_w, bev_h)
                # bev_prior_feat = bev_prior_feat.reshape(-1, bev_dim, bev_w, bev_h)
                # bev_prior_feat = bev_prior_feat + x
                if grid_encoder:

                    (pixel_cat_map_gt_clone, all_disp_field_gt_clone, motion_gt_clone) = disp_gt_map
                    all_disp_field_gt_clone = all_disp_field_gt_clone.view(bevs.shape[0], self.out_seq_len, bevs.shape[-2], bevs.shape[-1], -1)
                    gt_gird_feat = self.grid_prior(all_disp_field_gt_clone, pixel_cat_map_gt_clone, motion_gt_clone)
                    con_x = torch.cat([bev_prior_feat, gt_gird_feat, *disp_gt_map], dim=1)
                else:

                    con_x = torch.cat([bev_prior_feat, *disp_gt_map], dim=1)
            else:
                if grid_encoder:
                    (pixel_cat_map_gt_clone, all_disp_field_gt_clone, motion_gt_clone) = disp_gt_map
                    all_disp_field_gt_clone = all_disp_field_gt_clone.view(bevs.shape[0], self.out_seq_len, bevs.shape[-2], bevs.shape[-1], -1)
                    gt_gird_feat = self.grid_prior(all_disp_field_gt_clone, pixel_cat_map_gt_clone, motion_gt_clone)
                    con_x = torch.cat([x, gt_gird_feat, *disp_gt_map], dim=1)
                else:
                    con_x = torch.cat([x, *disp_gt_map], dim=1)
                # con_x = torch.cat([x, disp_gt_map], dim=1)
        else:
            if instance_info is not None and bev_prior_feat is not None:
                bev_prior_feat = bev_prior_feat.contiguous().view(-1, bev_dim, bev_w, bev_h)
                con_x = torch.cat([x, bev_prior_feat], dim=1)
            else:
                con_x = torch.cat([x, x], dim=1)
            # pass

        """gt bev latent distribution"""
        # bev_gt_refine, _, future_mu, future_log_sigma, _ = self.future_distrubution(con_x)
        future_distr_dict = self.future_distrubution(con_x, motion_masks)
        future_z_dynamic_distribution = future_distr_dict['dynamic'][0]

        future_mu_dyn, future_logvar_dyn = future_distr_dict['dynamic'][1], future_distr_dict['dynamic'][2]


        vae_dict = {
            'present_mu_dyn': present_mu_dyn,
            'present_log_sigma_dyn': present_logvar_dyn,
            'future_mu_dyn': future_mu_dyn,
            'future_log_sigma_dyn': future_logvar_dyn,
        }

        if future_z_dynamic_distribution is not None:
            z_dyn = future_z_dynamic_distribution

        # bev_refine: [bs, latent, h, w]
        # bev_refine: [bs, latent, 1, 1]
        # breakpoint()
        # latent=32
        _, latent_dim, gauss_h, gauss_w = z_dyn.shape
        # auto_z = z_dyn.unsqueeze(1).repeat(1, self.out_seq_len, 4, 4, 4)
        # disp_feat = self.future_prediction(auto_z, x_tiny)

        # bev_refine = bev_refine.expand(-1, x.size(-3), x.size(-2), x.size(-1)) + x
        """motion mask divided"""
        z_dyn = z_dyn.repeat(1, 1, int(256/gauss_h), int(256/gauss_w) ) + x


        # TODO: res connection
        """bev_refine=x + bev_refine"""
        # Cell Classification head
        cell_class_pred = self.cell_classify(z_dyn)

        # Motion State Classification head
        state_class_pred = self.state_classify(z_dyn)
        # Dual Motion Displacement prediction
        disp = self.motion_pred(z_dyn)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))


        disp_stat = self.motion_pred_stat(z_dyn)
        disp_stat = disp_stat.view(-1, 2, x.size(-2), x.size(-1))

        # data augmentation with direction
        # rotation disp list
        disp_rotated_list = []
        if dir_aug:
            # rotation angles
            angles = [90, 180, 270]
            disp_clone = disp.clone()
            disp_clone = disp.view(bs, -1, 2, x.size(-2), x.size(-1))
            T = disp_clone.shape[1]


            for angle in angles:
                # rotate bev
                rotated_bev_feature = rotate_tensor(disp_feat, angle)

                # decoder disp with rotation
                rotated_disp_pred = self.motion_pred(rotated_bev_feature)

                # rotated_disp_pred reshape
                rotated_disp_pred = rotated_disp_pred.view(bs, T, -1, x.size(-2), x.size(-1))

                # inverse rotation
                inv_rotated_disp_pred = inverse_rotate_tensor(rotated_disp_pred, angle)
                assert inv_rotated_disp_pred.shape == disp_clone.shape, f"Shapes do not match: {inv_rotated_disp_pred.shape} != {disp_gt.shape}"

                # inverse rotated disp pred shape
                inv_rotated_disp_pred = inv_rotated_disp_pred.view(-1, 2, x.size(-2), x.size(-1))
                disp_rotated_list.append(inv_rotated_disp_pred)

        return disp, disp_stat, cell_class_pred, state_class_pred, vae_dict, x


"""For VAE generator with spatialGRU with prior knowledge extractor with dual"""
class PriorMotionGridInsv2(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, num_grid=100):
        super(PriorMotionGridInsv2, self).__init__()
        self.out_seq_len = out_seq_len

        self.stpn = STPN(height_feat_size=height_feat_size)
        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=out_seq_len)
        self.motion_pred_stat = MotionPrediction(seq_len=out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)

        """prior info grid  prior"""
        # self.grid_prior = GridPrior()
        self.grid_prior = GridPriorv2(seq_len=out_seq_len, cell_input_dim=16, num_cls=5, num_state=motion_category_num)

        """prior info extractor"""
        # grid positional encoding
        self.grid_pos_encoding = SineContinuousPositionalEncoding(
                                num_feats=16,  # 2 * 16 = 32 final dim
                                temperature=1000,
                                normalize=True,
                                range=[128, 128],
                                offset=[128, 128],
                            )
        # instance encoder
        self.instance_attn = GridInsTransformer(
                            input_dim=32*num_grid + 32*out_seq_len + 5,  # 32 * 11 + 8
                            dmodel=32,      # TODO: maybe change?
                            hidden_dim=256,  # TODO: maybe change?
                            nheads=4,
                            nlayers=6,
                            batch_first=True,
                            pos_encoder="sin",
                            disp_encoder="temporal",
                        )

        self.bev_pos_encoding = SimplePositionEncoding(num_feats=16, row_num_embed=256, col_num_embed=256)
        self.bev_attn = MaskedCrossAttention(embed_dims=32,
                                            num_heads=8,
                                            dropout=0.2,
                                            init_cfg=None,
                                            batch_first=True
                                                ) # add layernorm?

        self.present_distrubution = DualBranchVAEv2(in_channels=32, latent_dim=32, hidden_dims=[32, 64, 128, 256],
                                    bev_size=256)

        self.future_distrubution = DualBranchVAEv2(in_channels=32+40+out_seq_len*2+5+2, latent_dim=32, hidden_dims=[32, 64, 128, 256],
                          bev_size=256)
        self.conv_fuser = ConvFuser(in_channels=[32, 32], out_channels=32)
        # self.future_prediction = FuturePrediction(in_channels=32*4, latent_dim=32*4)
    def forward(self, bevs, disp_gt_map=None, instance_info=None, dir_aug=False, grid_encoder=False, non_empty_map=None, motion_state=None,
                training_only=True):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # motion_gt_clone = disp_gt_map[-1]
        if motion_state.shape[1] != 2:
            motion_state = motion_state.permute(0, 3, 1, 2)
        motion_gt_clone = motion_state.to(bevs.device)
        motion_gt_state = torch.argmax(motion_gt_clone, dim=1) + 1
        if non_empty_map is not None:
            non_empty_map = non_empty_map.view(-1, bevs.shape[-2], bevs.shape[-1]).to(bevs.device)
            motion_gt_state = (non_empty_map * motion_gt_state).to(torch.int32)
        dyn_mask = (motion_gt_state==2)
        stat_mask = (motion_gt_state==1)

        motion_masks = (dyn_mask, stat_mask)
        # Backbone network
        x, x_tiny, _ = self.stpn(bevs)

        bs, bev_dim, bev_h, bev_w = x.size()

        bev_prior_feat = None

        """Instance information"""
        if instance_info is not None:
            # instance info extractor with self-attn
            (ins_pe_list, ins_disp_list, onehot_category_list) = instance_info
            break_flag = False
            for ins in ins_pe_list:
                if ins.shape[0] == 0 or ins.shape[0] == 1:
                    bev_prior_feat = None
                    break_flag = True
            if not break_flag:
                ins_feat = self.instance_attn(ins_pe_list, ins_disp_list, onehot_category_list, device=x.device)

                if isinstance(ins_feat, list):
                    # batch, num_polylines * feat_dim
                    ins_graph_shapes = torch.tensor([mfeat.shape[0] for mfeat in ins_feat],
                                                    device=ins_feat[0].device)

                    max_num_polylines = ins_graph_shapes.max()
                    # breakpoint()
                    ins_feat = torch.stack([torch.cat([mfeat, mfeat.new_zeros(max_num_polylines - mfeat.shape[0], mfeat.shape[1])], dim=0) for mfeat in ins_feat])  # batch, max_num_polylines, feat_dim
                bev_mask = torch.zeros((bs, bev_h, bev_w)).to(x.device)
                bev_pos = self.bev_pos_encoding(bev_mask)
                # breakpoint()
                bev_pos = bev_pos.flatten(2).permute(2, 0, 1).permute(1, 0, 2)

                bev = x.clone()
                bev = bev.view(-1, bev_dim, bev_w*bev_h).permute(0, 2, 1)
                bev_prior_feat = self.bev_attn(bev, ins_feat, ins_feat, query_pos=bev_pos, residual=bev, spatial_shapes=ins_graph_shapes)

        """present distribution
        {
            'dynamic': (z_dyn, mu_dyn, logvar_dyn),
            'static': (z_stat, mu_stat, logvar_stat),
            'input': input
        }"""
        # bev_refine, _, present_mu, present_log_sigma, _ = self.present_distrubution(x)
        present_distr_dict = self.present_distrubution(x, motion_masks)
        z_dyn = present_distr_dict['dynamic'][0]

        present_mu_dyn, present_logvar_dyn = present_distr_dict['dynamic'][1], present_distr_dict['dynamic'][2]


        # x_detach = x.detach()
        future_z_dynamic_distribution, vae_dict = None, None
        if disp_gt_map is not None:
            if instance_info is not None and bev_prior_feat is not None:
                bev_prior_feat = bev_prior_feat.contiguous().view(-1, bev_dim, bev_w, bev_h)
                # bev_prior_feat = bev_prior_feat.reshape(-1, bev_dim, bev_w, bev_h)
                # bev_prior_feat = bev_prior_feat + x
                if grid_encoder:

                    (pixel_cat_map_gt_clone, all_disp_field_gt_clone, motion_gt_clone) = disp_gt_map
                    all_disp_field_gt_clone = all_disp_field_gt_clone.view(bevs.shape[0], self.out_seq_len, bevs.shape[-2], bevs.shape[-1], -1)
                    gt_gird_feat = self.grid_prior(all_disp_field_gt_clone, pixel_cat_map_gt_clone, motion_gt_clone)
                    con_x = torch.cat([bev_prior_feat, gt_gird_feat, *disp_gt_map], dim=1)
                else:

                    con_x = torch.cat([bev_prior_feat, *disp_gt_map], dim=1)
            else:
                if grid_encoder:
                    (pixel_cat_map_gt_clone, all_disp_field_gt_clone, motion_gt_clone) = disp_gt_map
                    all_disp_field_gt_clone = all_disp_field_gt_clone.view(bevs.shape[0], self.out_seq_len, bevs.shape[-2], bevs.shape[-1], -1)
                    gt_gird_feat = self.grid_prior(all_disp_field_gt_clone, pixel_cat_map_gt_clone, motion_gt_clone)
                    con_x = torch.cat([x, gt_gird_feat, *disp_gt_map], dim=1)
                else:
                    con_x = torch.cat([x, *disp_gt_map], dim=1)
                # con_x = torch.cat([x, disp_gt_map], dim=1)
        else:
            if instance_info is not None and bev_prior_feat is not None:
                bev_prior_feat = bev_prior_feat.contiguous().view(-1, bev_dim, bev_w, bev_h)
                con_x = torch.cat([x, bev_prior_feat], dim=1)
            # else:
                con_x = torch.cat([x, x], dim=1)
            # pass
        if training_only:
            """gt bev latent distribution"""
            # bev_gt_refine, _, future_mu, future_log_sigma, _ = self.future_distrubution(con_x)
            future_distr_dict = self.future_distrubution(con_x, motion_masks)
            future_z_dynamic_distribution = future_distr_dict['dynamic'][0]

            future_mu_dyn, future_logvar_dyn = future_distr_dict['dynamic'][1], future_distr_dict['dynamic'][2]


            vae_dict = {
                'present_mu_dyn': present_mu_dyn,
                'present_log_sigma_dyn': present_logvar_dyn,
                'future_mu_dyn': future_mu_dyn,
                'future_log_sigma_dyn': future_logvar_dyn,
            }

        if future_z_dynamic_distribution is not None:
            z_dyn = future_z_dynamic_distribution
        # bev_refine: [bs, latent, h, w]
        # bev_refine: [bs, latent, 1, 1]
        # breakpoint()
        # latent=32
        _, latent_dim, gauss_h, gauss_w = z_dyn.shape
        # auto_z = z_dyn.unsqueeze(1).repeat(1, self.out_seq_len, 4, 4, 4)
        # disp_feat = self.future_prediction(auto_z, x_tiny)

        # bev_refine = bev_refine.expand(-1, x.size(-3), x.size(-2), x.size(-1)) + x
        """motion mask divided"""
        z_dyn = z_dyn.repeat(1, 1, int(256/gauss_h), int(256/gauss_w) ) + x

        # TODO: res connection
        """bev_refine=x + bev_refine"""
        # Cell Classification head
        cell_class_pred = self.cell_classify(z_dyn)

        # Motion State Classification head
        state_class_pred = self.state_classify(z_dyn)
        # Dual Motion Displacement prediction
        disp = self.motion_pred(z_dyn)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))


        disp_stat = self.motion_pred_stat(z_dyn)
        disp_stat = disp_stat.view(-1, 2, x.size(-2), x.size(-1))

        # data augmentation with direction
        # rotation disp list
        disp_rotated_list = []
        if dir_aug:
            # rotation angles
            angles = [90, 180, 270]
            disp_clone = disp.clone()
            disp_clone = disp.view(bs, -1, 2, x.size(-2), x.size(-1))
            T = disp_clone.shape[1]


            for angle in angles:
                # rotate bev
                rotated_bev_feature = rotate_tensor(disp_feat, angle)

                # decoder disp with rotation
                rotated_disp_pred = self.motion_pred(rotated_bev_feature)

                # rotated_disp_pred reshape
                rotated_disp_pred = rotated_disp_pred.view(bs, T, -1, x.size(-2), x.size(-1))

                # inverse rotation
                inv_rotated_disp_pred = inverse_rotate_tensor(rotated_disp_pred, angle)
                assert inv_rotated_disp_pred.shape == disp_clone.shape, f"Shapes do not match: {inv_rotated_disp_pred.shape} != {disp_gt.shape}"

                # inverse rotated disp pred shape
                inv_rotated_disp_pred = inv_rotated_disp_pred.view(-1, 2, x.size(-2), x.size(-1))
                disp_rotated_list.append(inv_rotated_disp_pred)

        return disp, disp_stat, cell_class_pred, state_class_pred, vae_dict, x

class PriorMotionDual(nn.Module):
    def __init__(self, out_seq_len=20, motion_category_num=2, height_feat_size=13, num_grid=100):
        super(PriorMotionDual, self).__init__()
        self.out_seq_len = out_seq_len

        self.stpn = STPN(height_feat_size=height_feat_size)
        self.cell_classify = CellClassification()
        self.motion_pred = MotionPredictionV2(seq_len=out_seq_len)
        self.motion_pred_stat = MotionPrediction(seq_len=out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)

        """prior info grid  prior"""
        # self.grid_prior = GridPrior()
        self.grid_prior = GridPriorv2(seq_len=out_seq_len, cell_input_dim=16, num_cls=5, num_state=motion_category_num)
        """prior info extractor"""
        # grid positional encoding
        self.grid_pos_encoding = SineContinuousPositionalEncoding(
                                num_feats=16,  # 2 * 16 = 32 final dim
                                temperature=1000,
                                normalize=True,
                                range=[128, 128],
                                offset=[128, 128],
                            )
        # instance encoder
        self.instance_attn = GridInsTransformer(
                            input_dim=32*num_grid + 32*out_seq_len + 5,  # 32 * 11 + 8
                            dmodel=32,      # TODO: maybe change?
                            hidden_dim=256,  # TODO: maybe change?
                            nheads=4,
                            nlayers=6,
                            batch_first=True,
                            pos_encoder="sin",
                            disp_encoder="temporal",
                        )

        self.bev_pos_encoding = SimplePositionEncoding(num_feats=16, row_num_embed=256, col_num_embed=256)
        self.bev_attn = MaskedCrossAttention(embed_dims=32,
                                            num_heads=8,
                                            dropout=0.2,
                                            init_cfg=None,
                                            batch_first=True
                                                ) # add layernorm?

        self.present_distrubution = SimpleVAEV2(in_channels=32, latent_dim=32, hidden_dims=[32, 64, 128, 256],
                                    bev_size=256)

        self.future_distrubution = SimpleVAEV2(in_channels=32+40+ 2*out_seq_len + 5 + 2, latent_dim=32, hidden_dims=[32, 64, 128, 256],
                          bev_size=256)
        self.conv_fuser = ConvFuser(in_channels=[32, 32], out_channels=32)
        self.future_prediction = FuturePrediction(in_channels=32*4, latent_dim=32*4)
    def forward(self, bevs, disp_gt_map=None, instance_info=None, dir_aug=False, grid_encoder=False, non_empty_map=None, \
                motion_state=None, training_only=True):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        # motion_gt_clone = disp_gt_map[-1]
        if motion_state.shape[1] != 2:
            motion_state = motion_state.permute(0, 3, 1, 2)
        # if motion_state.shape[1] != 2:
        motion_gt_clone = motion_state.to(bevs.device)
        motion_gt_state = torch.argmax(motion_gt_clone, dim=1) + 1
        if non_empty_map is not None:
            non_empty_map = non_empty_map.view(-1, bevs.shape[-2], bevs.shape[-1]).to(bevs.device)
            motion_gt_state = (non_empty_map * motion_gt_state).to(torch.int32)
        dyn_mask = (motion_gt_state==2)
        stat_mask = (motion_gt_state==1)

        motion_masks = (dyn_mask, stat_mask)
        # Backbone network
        x, x_tiny, _ = self.stpn(bevs)

        bs, bev_dim, bev_h, bev_w = x.size()

        bev_prior_feat = None

        """Instance information"""
        if instance_info is not None:
            # instance info extractor with self-attn
            (ins_pe_list, ins_disp_list, onehot_category_list) = instance_info
            break_flag = False
            for ins in ins_pe_list:
                if ins.shape[0] == 0 or ins.shape[0] == 1:
                    bev_prior_feat = None
                    break_flag = True
            if not break_flag:
                ins_feat = self.instance_attn(ins_pe_list, ins_disp_list, onehot_category_list, device=x.device)

                if isinstance(ins_feat, list):
                    # batch, num_polylines * feat_dim
                    ins_graph_shapes = torch.tensor([mfeat.shape[0] for mfeat in ins_feat],
                                                    device=ins_feat[0].device)

                    max_num_polylines = ins_graph_shapes.max()
                    # breakpoint()
                    ins_feat = torch.stack([torch.cat([mfeat, mfeat.new_zeros(max_num_polylines - mfeat.shape[0], mfeat.shape[1])], dim=0) for mfeat in ins_feat])  # batch, max_num_polylines, feat_dim
                bev_mask = torch.zeros((bs, bev_h, bev_w)).to(x.device)
                bev_pos = self.bev_pos_encoding(bev_mask)
                # breakpoint()
                bev_pos = bev_pos.flatten(2).permute(2, 0, 1).permute(1, 0, 2)

                bev = x.clone()
                bev = bev.view(-1, bev_dim, bev_w*bev_h).permute(0, 2, 1)
                bev_prior_feat = self.bev_attn(bev, ins_feat, ins_feat, query_pos=bev_pos, residual=bev, spatial_shapes=ins_graph_shapes)

        """present distribution
        {
            'dynamic': (z_dyn, mu_dyn, logvar_dyn),
            'static': (z_stat, mu_stat, logvar_stat),
            'input': input
        }"""
        prsent_z, _, present_mu, present_log_sigma, _ = self.present_distrubution(x)
        # present_distr_dict = self.present_distrubution(x, motion_masks)
        # z_dyn = present_distr_dict['dynamic'][0]

        # present_mu_dyn, present_logvar_dyn = present_distr_dict['dynamic'][1], present_distr_dict['dynamic'][2]


        # x_detach = x.detach()
        future_z, vae_dict = None, None
        if disp_gt_map is not None:
            if instance_info is not None and bev_prior_feat is not None:
                bev_prior_feat = bev_prior_feat.contiguous().view(-1, bev_dim, bev_w, bev_h)
                # bev_prior_feat = bev_prior_feat.reshape(-1, bev_dim, bev_w, bev_h)
                # bev_prior_feat = bev_prior_feat + x
                if grid_encoder:

                    (pixel_cat_map_gt_clone, all_disp_field_gt_clone, motion_gt_clone) = disp_gt_map
                    all_disp_field_gt_clone = all_disp_field_gt_clone.view(bevs.shape[0], self.out_seq_len, bevs.shape[-2], bevs.shape[-1], -1)
                    gt_gird_feat = self.grid_prior(all_disp_field_gt_clone, pixel_cat_map_gt_clone, motion_gt_clone)
                    con_x = torch.cat([bev_prior_feat, gt_gird_feat, *disp_gt_map], dim=1)
                else:

                    con_x = torch.cat([bev_prior_feat, *disp_gt_map], dim=1)
            else:
                if grid_encoder:
                    (pixel_cat_map_gt_clone, all_disp_field_gt_clone, motion_gt_clone) = disp_gt_map
                    all_disp_field_gt_clone = all_disp_field_gt_clone.view(bevs.shape[0], self.out_seq_len, bevs.shape[-2], bevs.shape[-1], -1)
                    gt_gird_feat = self.grid_prior(all_disp_field_gt_clone, pixel_cat_map_gt_clone, motion_gt_clone)
                    con_x = torch.cat([x, gt_gird_feat, *disp_gt_map], dim=1)
                else:
                    con_x = torch.cat([x, *disp_gt_map], dim=1)
                # con_x = torch.cat([x, disp_gt_map], dim=1)
        else:
            # con_x = torch.cat()
            pass
        if training_only:
            """gt bev latent distribution"""
            future_z, _, future_mu, future_log_sigma, _ = self.future_distrubution(con_x)



            vae_dict = {
                'present_mu': present_mu,
                'present_log_sigma': present_log_sigma,
                'future_mu': future_mu,
                'future_log_sigma': future_log_sigma,
            }


        if future_z is not None:
            prsent_z = future_z

        # bev_refine: [bs, latent, h, w]
        # bev_refine: [bs, latent, 1, 1]
        # breakpoint()
        # latent=32
        _, latent_dim, gauss_h, gauss_w = prsent_z.shape
        auto_z = prsent_z.unsqueeze(1).repeat(1, self.out_seq_len, 4, 4, 4)
        # breakpoint()
        disp_feat = self.future_prediction(auto_z, x_tiny)
        """motion mask divided"""
        prsent_z = prsent_z.repeat(1, 1, int(256/gauss_h), int(256/gauss_w) ) + x



        # TODO: res connection
        """bev_refine=x + bev_refine"""
        # Cell Classification head
        cell_class_pred = self.cell_classify(prsent_z)

        # Motion State Classification head
        state_class_pred = self.state_classify(prsent_z)
        # Dual Motion Displacement prediction
        disp = self.motion_pred(disp_feat)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))


        disp_stat = self.motion_pred_stat(prsent_z)
        disp_stat = disp_stat.view(-1, 2, x.size(-2), x.size(-1))

        # data augmentation with direction
        # rotation disp list
        disp_rotated_list = []
        if dir_aug:
            # rotation angles
            angles = [90, 180, 270]
            disp_clone = disp.clone()
            disp_clone = disp.view(bs, -1, 2, x.size(-2), x.size(-1))
            T = disp_clone.shape[1]


            for angle in angles:
                # rotate bev
                rotated_bev_feature = rotate_tensor(disp_feat, angle)

                # decoder disp with rotation
                rotated_disp_pred = self.motion_pred(rotated_bev_feature)

                # rotated_disp_pred reshape
                rotated_disp_pred = rotated_disp_pred.view(bs, T, -1, x.size(-2), x.size(-1))

                # inverse rotation
                inv_rotated_disp_pred = inverse_rotate_tensor(rotated_disp_pred, angle)
                assert inv_rotated_disp_pred.shape == disp_clone.shape, f"Shapes do not match: {inv_rotated_disp_pred.shape} != {disp_gt.shape}"

                # inverse rotated disp pred shape
                inv_rotated_disp_pred = inv_rotated_disp_pred.view(-1, 2, x.size(-2), x.size(-1))
                disp_rotated_list.append(inv_rotated_disp_pred)

        return disp, disp_stat, cell_class_pred, state_class_pred, vae_dict, x
