import sys
sys.path.append('/workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion')
from model.model import LegoMotion
from model.model import GridInsMotionNet
from model.LegoMotion import LEGOMotion
from model.prior_extractor import GridPriorv2
from model.PriorMotion import PriorMotion, PriorMotionIns
# from model.model import LegoMotion_lowresX
from model.vae_model import DualBranchVAE, DualBranchVAEv2
import torch
# MaskEncoder
v = torch.rand((16, 5, 256, 256, 13))
disp_gt, cls_gt, state_gt = torch.rand((16, 20, 256, 256, 2)), torch.rand((16, 5, 256, 256)), torch.rand((16, 2, 256, 256)),
disp_gt_map = (cls_gt, disp_gt, state_gt)
bev_feat = torch.rand((16, 32, 256, 256))

masks = torch.rand((16, 1, 256, 256))

# model = PriorMotionIns()

# model = GridPriorv2()
model = DualBranchVAEv2(in_channels=32, latent_dim=32, hidden_dims=[32, 64, 128, 256])
# model = LegoMotion_lowresX()
# model = GridInsMotionNet()
# output = model(disp_gt, cls_gt, state_gt)
output = model(bev_feat, (masks, masks))
# output = model(v, disp_gt_map=None, instance_info=None, dir_aug=False, grid_encoder=False, motion_state=state_gt)
breakpoint()
print(output.shape)
