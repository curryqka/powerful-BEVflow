# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

import torch
import torch.nn as nn
import numpy as np
import argparse
import os,sys
sys.path.append("/workspace/qiankangan/Motion/powerful-BEVflow-PriorMotion")
import warnings
from model.model import MotionNetMGDA, MotionNet
from model.en_model import En_MotionNet, GAN_MotionNet, GAN_MotionNetV2, En_GAN_MotionNet, En_GAN_MotionNetV2, MotionNetVAE, GRUMotionNetVAE, UniGRUMotionNetVAE
from model.en_gan_model import FeatEncoder, MotionNetGenerator
from model.model_ablation import MotionNet_MLP, MotionNet_CNN, MotionNetHigher, MotionNet_GRU
from model.model import GridInsMotionNet
# from model.be_sti import MotionNet
from utlis.model_utlis import *
from data.nuscenes_dataloader import DatasetSingleSeq
from sklearn.metrics import confusion_matrix

from time import *

model_path = "/home/qianka/Motion/logs/UniGRUVAE_MotionNet_test/train_multi_seq/2024-11-01_14-12-06/epoch_24.pth"
model_path = "/home/qianka/Motion/logs/MotionNet/train_multi_seq/2024-04-17_13-10-01/epoch_45.pth"
model_path = "/home/qianka/Motion/logs/VAE_MotionNet_test/train_multi_seq/2024-10-06_15-51-40/epoch_45.pth"
model_path = "/home/qianka/Motion/logs/GRUVAE_MotionNet_test/train_multi_seq/2024-10-19_11-56-46/epoch_45.pth"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda")

# model = UniGRUMotionNetVAE()
# model = MotionNet()
# model = MotionNetVAE()
model = GridInsMotionNet()
model = nn.DataParallel(model)


# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)

model.eval()


x = torch.rand(1, 5, 256, 256, 13).to(device)

t1 = time()
for i in range(1000):
    print(i)
    # y = model_encoder(x)
    z = model(x)

t2 = time()

runtime = t2-t1
print(runtime)
