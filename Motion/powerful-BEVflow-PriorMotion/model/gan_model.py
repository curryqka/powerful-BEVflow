import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model import Conv3D
class DiscriminatorInfo(nn.Module):
    r"""An Discriminator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1704.00028v3>`_ paper.
    """

    def __init__(self, in_channel=32, out_channel=128) -> None:
        super(DiscriminatorInfo, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, \
            kernel_size=3, stride=1, padding=1, bias=False)

        self.conv2 = nn.Conv2d(64, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.QHead = QHead()
        self.DHead = DHead()
        # self.conv3 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        # self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1 , inplace=True)

        # discriminator
        true_prob = self.DHead(x)

        # info
        disp, cell_class_pred, state_class_pred, mu, var = self.QHead(x)
        return disp, cell_class_pred, state_class_pred, true_prob, mu, var
class DiscriminatorAC(nn.Module):
    r"""An Discriminator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1704.00028v3>`_ paper.
    """
    def __init__(self, in_channel=32, out_channel=32) -> None:
        super(DiscriminatorAC, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, \
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)

        self.conv2 = nn.Conv2d(32, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)

        self.AHead = AHead()
        self.DHead = DHead()
        # self.conv3 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        # self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        # x shape: (bs, 32, w, h)
        res = x
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
        # x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1 , inplace=True)

        x = self.bn2(self.conv2(x))
        x = x + res
        x = F.leaky_relu(x, 0.1, inplace=True)
        # discriminator
        true_prob = self.DHead(x)

        # info
        disp, cell_class_pred, state_class_pred= self.AHead(x)
        return disp, cell_class_pred, state_class_pred, true_prob

class GeneratorAC(nn.Module):
    # TODO: get the input dim
    def __init__(self, input_channel=100, mid_channel=67, out_channel=32):
        super(GeneratorAC, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(input_channel, 512, 4, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False)

        self.tconv4 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)

        # self.tconv5 = nn.ConvTranspose2d(64, out_channel, 4, 2, padding=1, bias=False)

        self.conv = nn.Conv2d(mid_channel+64, out_channel, 3, 1, 1)
    def forward(self, x, labels: torch.tensor=None):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))

        # img = F.relu(self.tconv5(x))
        img = torch.concat([x, labels], dim=1)
        img = self.conv(img)
        return img

class DHead(nn.Module):
    def __init__(self, hidden_dim = 32,):
        super(DHead, self).__init__()

        self.conv = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, x):
        # output = torch.sigmoid(self.conv(x))
        # WGAN
        output = torch.flatten(self.conv(x))
        return output

class QHead(nn.Module):
    def __init__(self, hidden_dim = 32,out_seq_len=20, motion_category_num=2,) -> None:
        super(QHead, self).__init__()

        self.out_seq_len = out_seq_len
        self.conv1 = nn.Conv2d(hidden_dim, 32, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)

        self.conv_mu = nn.Conv2d(32, 2, 1)
        self.conv_var = nn.Conv2d(32, 2, 1)


    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
        
        # Cell Classification head
        cell_class_pred = self.cell_classify(x)

        # Motion State Classification head
        state_class_pred = self.state_classify(x)

        # Motion Displacement prediction
        disp = self.motion_pred(x)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disp, cell_class_pred, state_class_pred, mu, var

class AHead(nn.Module):
    def __init__(self, hidden_dim = 32, hidden_=False, out_seq_len=20, motion_category_num=2,) -> None:
        super(AHead, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.out_seq_len = out_seq_len
        
        if hidden_:
            self.conv1 = nn.Conv2d(hidden_dim, 32, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(32)

        self.cell_classify = CellClassification()
        self.motion_pred = MotionPrediction(seq_len=self.out_seq_len)
        self.state_classify = StateEstimation(motion_category_num=motion_category_num)

    def forward(self, x):
        if x.shape[1] != 32:
            x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
        
        # Cell Classification head
        cell_class_pred = self.cell_classify(x)

        # Motion State Classification head
        state_class_pred = self.state_classify(x)

        # Motion Displacement prediction
        disp = self.motion_pred(x)
        disp = disp.view(-1, 2, x.size(-2), x.size(-1))

        return disp, cell_class_pred, state_class_pred


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
    r""" An Discriminator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1610.09585>`_ paper.
    """
    def __init__(self, seq_len):
        super(MotionPrediction, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 2 * seq_len, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x

class DiscriminatorDisp(nn.Module):

    def __init__(self, in_channel=40, out_channel=512):
        super(DiscriminatorDisp, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, \
            kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)

        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=out_channel)

        self.AHead = AHead(hidden_dim=512, hidden_=True)
        self.DHead = DHead(hidden_dim=512)

    def forward(self, x: torch.Tensor):
        if len(x.shape) > 4:
            x = x.view(x.shape[0], -1, x.shape[2], x.shape[3])
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1 , inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1 , inplace=True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1 , inplace=True)
        # discriminator
        true_prob = self.DHead(x)

        disp, cell_class_pred, state_class_pred= self.AHead(x)

        return disp, cell_class_pred, state_class_pred, true_prob

class GeneratorDisp(nn.Module):
    r""" An Generator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1610.09585>`_ paper.
    """
    def __init__(self, grid_size:int=256, input_channel=100, mid_channel=38, out_channel=40) -> None:
        super().__init__()
    
        self.tconv1 = nn.ConvTranspose2d(input_channel, 512, 4, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False)

        self.tconv4 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)

        # self.tconv5 = nn.ConvTranspose2d(64, out_channel, 4, 2, padding=1, bias=False)

        self.conv = nn.Conv2d(mid_channel+64, out_channel, 3, 1, 1)

    def forward(self, x, labels: torch.tensor=None):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))

        # img = F.relu(self.tconv5(x))
        img = torch.concat([x, labels], dim=1)
        img = self.conv(img)
        out = img.reshape(img.shape[0], -1, img.shape[2], img.shape[3], 2)

        return out

class DiscriminatorSTPN(nn.Module):
    def __init__(self, height_feat_size=2):
        super(DiscriminatorSTPN, self).__init__()
        self.conv_pre_1 = nn.Conv2d(height_feat_size, 32, kernel_size=3, stride=1, padding=1)
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(64, 64, kernel_size=(9, 1, 1), stride=(4,1,1), padding=(0, 0, 0))
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

        self.AHead = AHead(hidden_dim=32)
        self.DHead = DHead(hidden_dim=32)

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

        if len(res_x.shape) > 4:
            res_x = res_x.view(x.shape[0], -1, x.shape[2], x.shape[3])

        # discriminator
        true_prob = self.DHead(res_x)

        disp, cell_class_pred, state_class_pred= self.AHead(res_x)

        return disp, cell_class_pred, state_class_pred, true_prob

class DiscriminatorSimple(nn.Module):
    """Implementation of Wasserstein GAN.
    """

    def __init__(self, motion_dim=(32, 256, 256), dim=32):
        super(DiscriminatorSimple, self).__init__()

        self.motion_dim = motion_dim[0]

        self.encoder = nn.Sequential(
            nn.Conv2d(self.motion_dim, dim, 4, 2, 1),
            nn.LeakyReLU(),
            # nn.ReLU(True),

            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.BatchNorm2d(2 * dim),
            nn.LeakyReLU(),
            # nn.ReLU(True),

            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.BatchNorm2d(4 * dim),
            nn.LeakyReLU(),
            # nn.ReLU(True),

            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.BatchNorm2d(8 * dim),
            nn.LeakyReLU(),
            # nn.ReLU(True),

            
            
        )

        # 4 kernel size with stride 2, i.e. half of size 
        # out: (8*dim, h/16, w/16)
        output_size = int(8*dim*(motion_dim[-2] / 16) * (motion_dim[-1] / 16))
        # print(output_size)
        self.discriminator_head = nn.Sequential(
            nn.Linear(output_size, 1),
            
            # nn.Sigmoid()
            nn.Sigmoid()
        )
        
    def forward(self, input:torch.Tensor, bev=None) -> torch.Tensor:
        if bev is not None:
            input = torch.concat([input, bev], dim=1)
        out = self.encoder(input)
        out = out.contiguous().view(-1, out.shape[-3]*out.shape[-2]*out.shape[-1])
        out = self.discriminator_head(out)

        return out

class DiscriminatorSimpleWGAN(nn.Module):
    """Implementation of Wasserstein GAN.
    """

    def __init__(self, motion_dim=(40, 256, 256), dim=32, with_bev=False):
        super(DiscriminatorSimpleWGAN, self).__init__()

        self.motion_dim = motion_dim[0]

        self.encoder = nn.Sequential(
            nn.Conv2d(self.motion_dim, dim, 4, 2, 1),
            nn.LeakyReLU(),
            # nn.ReLU(True),

            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            # nn.BatchNorm2d(2 * dim),
            nn.LeakyReLU(),
            # nn.ReLU(True),

            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            # nn.BatchNorm2d(4 * dim),
            nn.LeakyReLU(),
            # nn.ReLU(True),

            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            # nn.BatchNorm2d(8 * dim),
            nn.LeakyReLU(),
            # nn.ReLU(True),

            
            
        )

        # 4 kernel size with stride 2, i.e. half of size 
        # out: (8*dim, h/16, w/16)
        output_size = int(8*dim*(motion_dim[-2] / 16) * (motion_dim[-1] / 16))
        # print(output_size)
        self.discriminator_head = nn.Sequential(
            nn.Linear(output_size, 1),
            
            # nn.Sigmoid()
            # nn.Sigmoid()
        )
        
    def forward(self, input:torch.Tensor, bev=None) -> torch.Tensor:
        if bev is not None:
            input = torch.concat([input, bev], dim=1)
        out = self.encoder(input)
        out = out.contiguous().view(-1, out.shape[-3]*out.shape[-2]*out.shape[-1])
        out = self.discriminator_head(out)

        return out

class NoiseGenerator(torch.nn.Module):
    def __init__(self, channels_noise=100, channels_out=32, features_g=16):
        super(NoiseGenerator, self).__init__()
        self.gen = torch.nn.Sequential(
            #imgsize: 4 x 4
            self._block(in_channels = channels_noise,out_channels = features_g * 16,kernel_size = (4,4),
                        stride=(2,2),padding=1),
            # imgsize: 8 x 8
            self._block(in_channels=features_g * 16, out_channels=features_g * 8, kernel_size=(4, 4), stride=(2,2),
                        padding=1),
            # imgsize: 16 x 16
            self._block(in_channels=features_g * 8, out_channels=features_g * 4, kernel_size=(4, 4), stride=(2,2),
                        padding=1),
            # imgsize: 32 x 32
            self._block(in_channels=features_g * 4, out_channels=features_g * 2, kernel_size=(4, 4), stride=(2,2),
                        padding=1),
            # imgsize: N x 3 x 64 x 64
            torch.nn.ConvTranspose2d(
                in_channels=features_g * 2, out_channels=channels_out, kernel_size=(4,4), stride=(2,2),
                padding=(1,1)
            ),
            
        )

    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False
            ),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.ReLU()
        )
        return self.conv

    def forward(self,input):
        x = self.gen(input)
        return x