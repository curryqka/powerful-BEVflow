import torch
from pytorch_model_summary import summary
import sys
sys.path.append('/workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion')
from model.model import MotionNet, GridInsMotionNet
from model.en_model import UniGridInsGRUMotionNetVAE, UniGRUMotionNetVAE, GRUMotionNetVAE
from model.model_ablation import MotionNet_GRU, MotionNet_CNN, MotionNet_MLP, MotionNetHigher, UniGridGenMotionNet, GRUMotionNetVAEV2
from model.be_sti import GeneralMotionNet
# load model
out_seq_len=20
height_feat_size=13
print('==> Loading model..')
breakpoint()
# model = MotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
model = GridInsMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
dummy_input = torch.randn(1, 5, 256, 256, 13)
print(summary(model, dummy_input, show_input=False, show_hierarchical=False))
