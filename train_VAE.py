# Copyright (c) 2024 qka
# Author: qiankangan
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse,os,sys,time
import numpy as np
from shutil import copytree, copy
from sklearn.metrics import confusion_matrix
import torch.utils
from model.model import MotionNet, VAE_GeneralMotionNet
from model.en_model import En_MotionNet, GAN_MotionNet, GAN_MotionNetV2, En_GAN_MotionNet, En_GAN_MotionNetV2, MotionNetVAE, GRUMotionNetVAE, UniGRUMotionNetVAE, UniGridInsGRUMotionNetVAE
from model.gan_model import DiscriminatorSimple, DiscriminatorSimpleWGAN
from model.be_sti import be_stiVAE, VAE_Generalbe_sti
from data.nuscenes_dataloader import TrainDatasetMultiSeq, DatasetSingleSeq
from data.instance_extractor import get_instance_infoV2, get_instance_infoV3
from utlis.loss_utlis import choose_disp_weighted_map
from utlis.params_utlis import params
from utlis.model_utlis import *

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)
    
def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

 # set the params
voxel_size = params['voxel_size']
area_extents = params['area_extents']
H, W = params['grid_h'], params['grid_w']

use_weighted_loss = True  # Whether to set different weights for different grid cell categories for loss computation

pred_adj_frame_distance = True  # Whether to predict the relative offset between frames

height_feat_size = params['height_feat']  # The size along the height dimension
cell_category_num = params['cell_category_num']  # The number of object categories (including the background)

trans_matrix_idx = 1  # Among N transformation matrices (N=2 in our experiment), which matrix is used for alignment (see paper)
NOISE_FACTOR = 1

parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='MotionNet', type=str, help="The experiment name")
parser.add_argument('-d', '--data', default=None, type=str, help='The path to the preprocessed sparse BEV training data')
parser.add_argument('-t', '--evaldata', default='/path_to/nuScenes/input-data/val/', type=str, help='The path to the preprocessed sparse BEV validate data')

parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
parser.add_argument('--batch', default=8, type=int, help='Batch size')
parser.add_argument('--nepoch', default=45, type=int, help='Number of epochs')
parser.add_argument('--nworker', default=8, type=int, help='Number of workers')
parser.add_argument('--lr', default=0.0016, type=float, help='learning rate')
parser.add_argument('--pre_model_path', type=str, default=None, \
                    help='The path to pretrained backbone model')
parser.add_argument('--model', default="MotionNet", type=str, help="choose model[MotionNet, ...]")
parser.add_argument('--use_adj_frame_pred', action='store_true', help='Whether to use adjacent frame prediction')
parser.add_argument('--auxiliary', action='store_true', help='Whether to use auxiliary loss')
parser.add_argument('--bev', action="store_true", help='whether to add bev feature in the discriminator')
parser.add_argument('--use_dir_aug', action='store_true', help='Whether to use direction augmentation')

parser.add_argument('--reg_weight_bg_tc', default=0.1, type=float, help='Weight of background temporal consistency term')
parser.add_argument('--reg_weight_fg_tc', default=2.5, type=float, help='Weight of instance temporal consistency')
parser.add_argument('--reg_weight_sc', default=15.0, type=float, help='Weight of spatial consistency term')

parser.add_argument('--use_bg_tc', action='store_true', help='Whether to use background temporal consistency loss')
parser.add_argument('--use_fg_tc', action='store_true', help='Whether to use foreground loss in st.')
parser.add_argument('--use_sc', action='store_true', help='Whether to use spatial consistency loss')
parser.add_argument('--use_dw', action='store_true', help='Whether to use spatial consistency loss')
parser.add_argument('--dw_mode', type=str, default='motion_status', help='[motion_status, norm1, norm2, softmax, cls]')

# model
parser.add_argument('--instance_mode', type=str, default='ins', help='the ins info extractor')
parser.add_argument('--nn_sampling', action='store_true', help='Whether to use nearest neighbor sampling in bg_tc loss')
parser.add_argument('--log', action='store_true', help='Whether to log')
parser.add_argument('--logpath', default='', help='The path to the output log file')

parser.add_argument('--gpu', default='0')

parser.add_argument('--general', action='store_true', help='Whether to mask non general')
parser.add_argument('--sensor_type', type=str, default='nuscenes', help='The benchmark used')

# model
parser.add_argument("--grid_prior", action="store_true", help="Whether to use prior grid")
parser.add_argument("--use_v", action="store_true", help="Whether to use prior velocity")
parser.add_argument("--use_vdir", action="store_true", help="Whether to use prior velocity direction")
parser.add_argument("--use_lastgt", action="store_true", help="Whether to choose only the last disp map!")

parser.add_argument('--debug', action='store_true', help='Whether to choose debug')
args = parser.parse_args()
print(args)

need_log = args.log
BATCH_SIZE = args.batch
num_epochs = args.nepoch
num_workers = args.nworker
# qka add the gpu number 20240314
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" #指定只可见0号物理卡和1号物理卡，可省略
# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#指定0号逻辑卡为device

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True
# torch.cuda.set_device('cuda:{}'.format(0))
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = True
reg_weight_bg_tc = args.reg_weight_bg_tc  # The weight of background temporal consistency term
reg_weight_fg_tc = args.reg_weight_fg_tc  # The weight of foreground temporal consistency term
reg_weight_sc = args.reg_weight_sc  # The weight of spatial consistency term

use_bg_temporal_consistency = args.use_bg_tc
use_fg_temporal_consistency = args.use_fg_tc
use_spatial_consistency = args.use_sc
use_disp_weighted_loss = args.use_dw

use_nn_sampling = args.nn_sampling

get_general = args.general
out_seq_len = params['out_seq_len'][args.sensor_type]  # The number of future frames we are going to predict
exp_setting = f"exp_{args.exp}_bs_{BATCH_SIZE}"
def main():
    start_epoch = 1
    # Whether to log the training info
    if need_log:
        logger_root = args.logpath if args.logpath !="" else 'logs'
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        if args.resume == "":
            model_save_path = check_folder(logger_root)
            model_save_path = check_folder(os.path.join(model_save_path, 'train_multi_seq'))
            model_save_path = check_folder(os.path.join(model_save_path, time_stamp))

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "w")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.write("The experiment: {}\n".format(exp_setting))

            # logging the details for this exp
            saver.write("Command line: {}\n".format(" ".join(sys.argv[0:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()

            # Copy the code files as logs
            # copytree('nuscenes-devkit', os.path.join(model_save_path, 'nuscenes-devkit'))
            # copytree('data', os.path.join(model_save_path, 'data'))
            copytree('utlis', os.path.join(model_save_path, 'utlis'))
            python_files = [f for f in os.listdir('.') if f.endswith('.py')]
            for f in python_files:
                copy(f, model_save_path)
        else:
            model_save_path = args.resume  # eg, "logs/train_multi_seq/1234-56-78-11-22-33"

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "a")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()

        # Specify gpu device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_num = torch.cuda.device_count()
        print("device number", device_num)

        if args.sensor_type == 'wt128_0':
            use_velocity_flag = True
        else:
            use_velocity_flag = False
        # set dataset
        # trainset
        # breakpoint()
        trainset = TrainDatasetMultiSeq(dataset_root=args.data, future_frame_skip=0, voxel_size=voxel_size, area_extents=area_extents,
                                        num_category=cell_category_num, num_future_frames=out_seq_len,
                                        use_velocity_flag=use_velocity_flag,
                                        sensor_type=args.sensor_type, debug_mode=args.debug)
        trainloader = torch.utils.data.DataLoader(trainset, BATCH_SIZE, shuffle=True, num_workers=num_workers)
        print("Training dataset size: {}".format(len(trainset)))
        saver.write("Training dataset size: {}".format(len(trainset)))
        saver.flush()

        # evalset
        tmp = args.evaldata
        evalset_split = tmp.split('/')[-1] if tmp.split('/')[-1] != "" else tmp.split('/')[-2]
        evalset = DatasetSingleSeq(dataset_root=args.evaldata, split=evalset_split, future_frame_skip=0,
                                    use_velocity_flag=use_velocity_flag, num_future_frames=out_seq_len,
                                    voxel_size=voxel_size, area_extents=area_extents, num_category=cell_category_num,
                                    sensor_type=args.sensor_type, debug_mode=args.debug)
        

        evalloader = torch.utils.data.DataLoader(evalset, batch_size=1, shuffle=False, num_workers=num_workers)
        print("Eval dataset size: {}".format(len(evalset)))
        saver.write("Eval dataset size: {}".format(len(evalset)))
        saver.flush()

        # set model
        model_dict = {
            "MotionNet": MotionNet,
            "En_MotionNet": En_MotionNet,
            # "GAN_MotionNet": En_GAN_MotionNet,
            # TODO:model design combination
            "VAE_MotionNet": MotionNetVAE,
            "VAE_be-sti": be_stiVAE,
            "GRUVAE_MotionNet": GRUMotionNetVAE,
            "UniGRUVAE_MotionNet": UniGRUMotionNetVAE,
            "UniGridInsGRUMotionNetVAE": UniGridInsGRUMotionNetVAE,
            "VAE_GeneralBESTI": VAE_Generalbe_sti,
            "VAE_GeneralMotionNet": VAE_GeneralMotionNet,
            # "VAE_MixGeneralMotionNet": VAE_MixGeneralMotionNet,
        }

        assert args.model in model_dict.keys(), "The model name only support MotionNet and ..."
        # if args.use_v:
        # model = model_dict[args.model](out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        if args.model == "VAE_MotionNet" or args.model == "VAE_be-sti":
            model = model_dict[args.model](out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        elif args.model == "GRUVAE_MotionNet":
            model = GRUMotionNetVAE(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        elif args.model == "En_MotionNet":
            model = En_MotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        elif args.model == "UniGRUVAE_MotionNet":
            model = UniGRUMotionNetVAE(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        elif args.model == "UniGridInsGRUMotionNetVAE":
            model = UniGridInsGRUMotionNetVAE(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        elif args.model == "VAE_GeneralMotionNet":
            if args.use_v:
                if args.use_vdir:
                    model = VAE_GeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 2)
                else:
                    model = VAE_GeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 1)
            else:
                model = VAE_GeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        elif args.model == "VAE_GeneralBESTI":
            if args.use_v:
                if args.use_vdir:
                    model = VAE_Generalbe_sti(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 2)
                else:
                    model = VAE_Generalbe_sti(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size + 1)
            else:
                model = VAE_Generalbe_sti(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size)
        elif args.model == "VAE_MixGeneralMotionNet":
            if args.use_v:
                if args.use_vdir:
                    model = MixGeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size,
                                            with_direction=True)
                else:
                    model = MixGeneralMotionNet(out_seq_len=out_seq_len, motion_category_num=2, height_feat_size=height_feat_size,
                                            with_direction=False)

        # load the pre-trained WeakMotionNet
        pretrain_model_path = args.pre_model_path
        if pretrain_model_path:
            pre_model = torch.load(pretrain_model_path)
            new_model_dict = model.state_dict()
            for k,v in list(pre_model['model_state_dict'].items()):
                pre_model['model_state_dict']['.'.join(k.split('.')[1:])] =   pre_model['model_state_dict'].pop(k)
            state_dict = {k:v for k, v in pre_model['model_state_dict'].items() if k in new_model_dict.keys() and 'motion' not in k}
            new_model_dict.update(state_dict)
            model.load_state_dict(new_model_dict)

        # if device_num > 1:
            
        model = nn.DataParallel(model)
        model = model.to(device)
        
        # set loss function criterion
        if use_weighted_loss:
            criterion = nn.SmoothL1Loss(reduction='none')
        else:
            criterion = nn.SmoothL1Loss(reduction='sum')

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10,20,30,40], gamma=0.5)

        if args.resume != "":
            model_path = f"{args.resume}/latest.pth"
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            print("Load model from {}, at epoch {}".format(args.resume, start_epoch - 1))
        # init eval metric
        me_static, me_slow, me_fast, pixel_acc = 1e3, 1e3, 1e3, 1e3
        best_metric = me_static + me_fast
        for epoch in range(start_epoch, num_epochs + 1):
            lr = optimizer.param_groups[0]['lr']
            print("Epoch {}, learning rate {}".format(epoch, lr))

            if need_log:
                saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
                saver.flush()

            scheduler.step()
            model.train()

            # if 'GAN' in args.model:
            #     schedulerD.step()
            #     discriminator_model.train()

            # trainloop
            if args.model == "UniGRUVAE_MotionNet" or args.model == "UniGridInsGRUMotionNetVAE":
                loss_D, loss_G, loss_bg_tc, loss_sc, loss_fg_tc, \
                aux_loss_disp, aux_loss_class, aux_loss_motion, kld_loss \
                    = train_uni(model, criterion, trainloader, optimizer=optimizer, device=device, epoch=epoch)
            else:
                loss_D, loss_G, loss_bg_tc, loss_sc, loss_fg_tc, \
                aux_loss_disp, aux_loss_class, aux_loss_motion, kld_loss \
                    = train(model, criterion, trainloader, optimizer=optimizer, device=device, epoch=epoch)

            # evalloop
            model.eval()
        
            
            me_static, me_slow, me_fast, quantile_50_static, quantile_50_slow, quantile_50_fast, \
                    cat_map, mean_cat, pixel_acc = \
                eval(model=model, D=None, dataloader=evalloader, dataset=evalset, device=device, num_future_sweeps=out_seq_len,)
            
            if need_log:
                saver.write("{}\t{}\t{}\t{}\n\t{}\t{}\t{}\t{}\t{}\n".format(loss_D, loss_G, loss_bg_tc,
                                                            loss_fg_tc, loss_sc, \
                                                            aux_loss_disp, aux_loss_class, aux_loss_motion, \
                                                            kld_loss
                                                        ))
                saver.write("me_static: {}\t me_slow: {}\t me_fast: {}\n".format(me_static, me_slow, me_fast))
                saver.write("quantile_50_static: {}\t quantile_50_slow: {}\t quantile_50_fast: {}\n".format(quantile_50_static, quantile_50_slow, quantile_50_fast))
                saver.write("pixel acc: {}\n".format(pixel_acc))
                saver.flush()


            # save model
            """save the best models"""
            new_metric = me_static + me_fast
            if new_metric < best_metric:
                best_metric = new_metric
                if need_log:
                    save_dict = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": aux_loss_disp.avg,
                }
                torch.save(save_dict, os.path.join(model_save_path, "save_model.pth"))

            if need_log and (epoch % 5 == 0 or epoch == 1 or epoch > 20):
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': aux_loss_disp.avg,
                    # 'discriminator_model_state_dict': discriminator_model.state_dict(),
                    # 'optimizerD_state_dict': optimD.state_dict(),
                    # 'schedulerD_state_dict': schedulerD.state_dict(),
                    
                }

                cur_model_path = os.path.join(model_save_path, 'epoch_%s.pth'% str(epoch))
                torch.save(save_dict, cur_model_path)
                latest_model_path = os.path.join(model_save_path, 'latest.pth')
                if os.path.exists(latest_model_path):
                    os.system('rm {}'.format(latest_model_path))

                    os.symlink(cur_model_path, latest_model_path)
                else:
                    os.symlink(cur_model_path, latest_model_path)

    if need_log:
        print("log saved in {}".format(log_file_name))
        saver.close()

def train(model, criterion, dataloader, optimizer, optimD=None, device=None, epoch=1):
    # main generator
    running_loss_bg_tc = AverageMeter('bg_tc', ':.7f')  # background temporal consistency error
    running_loss_fg_tc = AverageMeter('fg_tc', ':.7f')  # foreground temporal consistency error
    running_loss_sc = AverageMeter('sc', ':.7f')  # spatial consistency error
    running_loss_D = AverageMeter('Discriminator', ':.7f')  # Discriminator error
    running_loss_G = AverageMeter('Generator', ':.7f')  # Generator error
    running_loss_kld =  AverageMeter('KLD', ':.6f')  # kld error
    # aux loss
    running_aux_loss_disp = AverageMeter('aux_Disp', ':.6f')  # for motion prediction error
    running_aux_loss_class = AverageMeter('aux_Obj_Cls', ':.6f')  # for cell classification error
    running_aux_loss_motion = AverageMeter('aux_Motion_Cls', ':.6f')  # for state estimation error

    # factor = (epoch / num_epochs) 
    for i, data in enumerate(dataloader, 0):
        # TODO: add FMCW output
        if args.sensor_type == 'nuscenes' or args.sensor_type == 'at128_0':
            padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, non_empty_map, pixel_cat_map_gt, \
                trans_matrices, motion_gt, pixel_instance_map, num_past_frames, num_future_frames = data
        
        elif args.sensor_type == 'wt128_0':
            (
            padded_voxel_points,
            all_disp_field_gt,
            all_valid_pixel_maps,
            non_empty_map,
            pixel_cat_map_gt,
            trans_matrices,
            motion_gt,
            pixel_instance_map,
            all_pixel_velocity_maps,
            all_pixel_direction_maps,
            all_pixel_cos_maps,
            num_past_frames,
            num_future_frames,
                ) = data
            all_pixel_velocity_maps = all_pixel_velocity_maps.view(-1, num_past_frames[0].item(), H, W, 1).to(device)
            all_pixel_direction_maps = all_pixel_direction_maps.view(-1, num_past_frames[0].item(), H, W, 1).to(device)
            all_pixel_cos_maps = all_pixel_cos_maps.view(-1, num_past_frames[0].item(), H, W, 3).to(device)


        # Move to GPU/CPU
        padded_voxel_points = padded_voxel_points.view(-1, num_past_frames[0].item(), H, W, height_feat_size)
        padded_voxel_points = padded_voxel_points.to(device)

        # reshape gt disp

        # all_disp_field_gt_clone = all_disp_field_gt.numpy()
        all_disp_field_gt_clone = all_disp_field_gt.clone().detach()
        future_steps = num_future_frames.numpy()[0]
        # print(all_disp_field_gt_clone.shape)
        # breakpoint()
        # reshape the valid pixel maps
        all_valid_pixel_maps_clone = all_valid_pixel_maps.clone().detach().view(-1, future_steps, H, W)
        valid_pixel_maps = all_valid_pixel_maps_clone[:, -future_steps:, ...].contiguous()
        valid_pixel_maps = torch.unsqueeze(valid_pixel_maps, 2)
        # print(valid_pixel_maps.shape)
        valid_pixel_maps = valid_pixel_maps.to(device=device)

        all_disp_field_gt_clone = all_disp_field_gt_clone.view(-1, future_steps, H, W, 2)
        all_disp_field_gt_clone = all_disp_field_gt_clone[:, -future_steps:, ...].contiguous()
        all_disp_field_gt_clone = torch.permute(all_disp_field_gt_clone, (0,1,4,2,3)).contiguous().to(device)

        if args.use_lastgt:
            all_disp_field_gt_clone = reshape_dispV2(all_disp_field_gt_clone, valid_pixel_maps, future_frames_num=future_steps, use_lastgt=True)
        else:
            all_disp_field_gt_clone = reshape_dispV3(all_disp_field_gt_clone, valid_pixel_maps, future_frames_num=future_steps, gt_flag=True)

        # pixel reshape
        pixel_cat_map_gt_clone = pixel_cat_map_gt.clone().detach()
        pixel_cat_map_gt_clone = pixel_cat_map_gt_clone.view(-1, cell_category_num, H, W).to(device)
        # motion reshape
        motion_gt_clone = motion_gt.clone().detach()
        motion_gt_clone = motion_gt_clone.view(-1, 2, H, W).to(device)

        # noise sample
        # breakpoint()
        scene_gt = torch.cat([pixel_cat_map_gt_clone, all_disp_field_gt_clone, motion_gt_clone], dim=1)
        # breakpoint()
        cell_gt = pixel_cat_map_gt_clone

        if args.sensor_type == 'nuscenes' or args.sensor_type == 'at128_0':
            disp_pred, class_pred, motion_pred, vae_dict = model(padded_voxel_points, scene_gt)
        elif args.sensor_type == 'wt128_0':
            if args.use_v:
                if args.use_vdir:
                    disp_pred, class_pred, motion_pred, vae_dict = model(padded_voxel_points, scene_gt, \
                                                                all_pixel_velocity_maps, all_pixel_direction_maps)
                else:
                    disp_pred, class_pred, motion_pred, vae_dict = model(padded_voxel_points, scene_gt, \
                                                                all_pixel_velocity_maps, None)
            else:
                disp_pred, class_pred, motion_pred, vae_dict = model(padded_voxel_points, scene_gt)
        
        

    
        optimizer.zero_grad()
        # compute reconstruct loss
        loss_all, loss_disp, loss_class, loss_motion, loss_bg_tc, loss_sc, loss_fg_tc = \
                compute_aux_loss(optimizer, device, num_future_frames[0].item(), all_disp_field_gt, all_valid_pixel_maps,
                            pixel_cat_map_gt, disp_pred, criterion, non_empty_map, class_pred, motion_gt,
                            motion_pred, trans_matrices, pixel_instance_map)
        
        # compute vae kld loss
        vae_loss = compute_vae_loss(vae_dict)
        re_weight = 5.0
        kld_weight = 0.5
        loss = re_weight * loss_all + kld_weight * vae_loss
        loss.backward()
        optimizer.step()


        if not all((loss_disp, loss_class, loss_motion)):
            print("{}, \t{}, \tat epoch {}, \titerations {} [empty occupy map]".
                  format(running_aux_loss_disp, running_aux_loss_class, epoch, i))
            continue

        running_loss_bg_tc.update(loss_bg_tc)
        running_loss_fg_tc.update(loss_fg_tc)
        running_loss_sc.update(loss_sc)
        running_aux_loss_disp.update(loss_disp)
        running_aux_loss_class.update(loss_class)
        running_aux_loss_motion.update(loss_motion)
        running_loss_D.update(-1)
        running_loss_G.update(-1)
        running_loss_kld.update(vae_loss.item())
            
        if i % 10 == 0:
            print("[{}/{}]\t{}, \t{}, \t{}, \t{}, \t{}, \n\t{}, \t{} \t{}, \n\t{}".
                format(epoch, i, running_loss_D, running_loss_G, \
                       running_aux_loss_disp, running_aux_loss_class, running_aux_loss_motion,
                       running_loss_bg_tc,
                     running_loss_sc, running_loss_fg_tc, running_loss_kld))
    return running_loss_D, running_loss_G, running_loss_bg_tc, running_loss_sc, running_loss_fg_tc, \
           running_aux_loss_disp, running_aux_loss_class, running_aux_loss_motion, running_loss_kld

def train_uni(model, criterion, dataloader, optimizer, optimD=None, device=None, epoch=1):
    # main generator
    running_loss_bg_tc = AverageMeter('bg_tc', ':.7f')  # background temporal consistency error
    running_loss_fg_tc = AverageMeter('fg_tc', ':.7f')  # foreground temporal consistency error
    running_loss_sc = AverageMeter('sc', ':.7f')  # spatial consistency error
    running_loss_D = AverageMeter('Discriminator', ':.7f')  # Discriminator error
    running_loss_G = AverageMeter('Generator', ':.7f')  # Generator error
    running_loss_kld =  AverageMeter('KLD', ':.6f')  # kld error
    # aux loss
    running_aux_loss_disp = AverageMeter('aux_Disp', ':.6f')  # for motion prediction error
    running_aux_loss_class = AverageMeter('aux_Obj_Cls', ':.6f')  # for cell classification error
    running_aux_loss_motion = AverageMeter('aux_Motion_Cls', ':.6f')  # for state estimation error

    # factor = (epoch / num_epochs) 
    for i, data in enumerate(dataloader, 0):
        
        padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, non_empty_map, pixel_cat_map_gt, \
            trans_matrices, motion_gt, pixel_instance_map, num_past_frames, num_future_frames = data

        # Move to GPU/CPU
        padded_voxel_points = padded_voxel_points.view(-1, num_past_frames[0].item(), H, W, height_feat_size)
        padded_voxel_points = padded_voxel_points.to(device)

        # reshape gt disp

        # all_disp_field_gt_clone = all_disp_field_gt.numpy()
        all_disp_field_gt_clone = all_disp_field_gt.clone().detach()
        future_steps = num_future_frames.numpy()[0]
        # print(all_disp_field_gt_clone.shape)

        # get numpy array
        all_disp_field_gt_numpy = all_disp_field_gt.clone().numpy()
        all_disp_field_gt_numpy = all_disp_field_gt_numpy.reshape(-1, future_steps, H, W, 2)
        all_disp_field_gt_numpy = all_disp_field_gt_numpy[:, -future_steps:,...]
        all_disp_field_gt_numpy = all_disp_field_gt_numpy.transpose(0, 1, 4, 2, 3)

        pixel_cat_map_gt_numpy = pixel_cat_map_gt.clone().numpy()
        pixel_cat_map_gt_numpy = pixel_cat_map_gt_numpy.reshape(-1, cell_category_num, H, W)
        
        all_valid_pixel_maps_numpy = all_valid_pixel_maps.clone().numpy()
        all_valid_pixel_maps_numpy = all_valid_pixel_maps_numpy.reshape(-1, future_steps, H, W)
        all_valid_pixel_maps_numpy = all_valid_pixel_maps_numpy[:, -future_steps:,...]
        all_valid_pixel_maps_numpy = all_valid_pixel_maps_numpy[:, :, None, ...]

        pixel_instance_map_gt_numpy = pixel_instance_map.clone().numpy()
        pixel_instance_map_gt_numpy = pixel_instance_map_gt_numpy.reshape(-1, H, W)

        # reshape the valid pixel maps
        all_valid_pixel_maps_clone = all_valid_pixel_maps.clone().detach().view(-1, future_steps, H, W)
        valid_pixel_maps = all_valid_pixel_maps_clone[:, -future_steps:, ...].contiguous()
        valid_pixel_maps = torch.unsqueeze(valid_pixel_maps, 2)
        valid_pixel_maps = valid_pixel_maps.to(device=device)

        all_disp_field_gt_clone = all_disp_field_gt_clone.view(-1, future_steps, H, W, 2)
        all_disp_field_gt_clone = all_disp_field_gt_clone[:, -future_steps:, ...].contiguous()
        all_disp_field_gt_clone = torch.permute(all_disp_field_gt_clone, (0,1,4,2,3)).contiguous().to(device)
        all_disp_field_gt_clone = reshape_dispV3(all_disp_field_gt_clone, valid_pixel_maps, gt_flag=True)

        # pixel reshape
        pixel_cat_map_gt_clone = pixel_cat_map_gt.clone().detach()
        pixel_cat_map_gt_clone = pixel_cat_map_gt_clone.view(-1, cell_category_num, H, W).to(device)
        # motion reshape
        motion_gt_clone = motion_gt.clone().detach()
        motion_gt_clone = motion_gt_clone.view(-1, 2, H, W).to(device)

        # pixel instance reshape
        pixel_instance_map_gt_clone = pixel_instance_map.clone().detach()
        pixel_instance_map_gt_clone = pixel_instance_map_gt_clone.view(-1, H, W)
        # breakpoint()
        if args.instance_mode == "ins":
            instance_positions, instance_displacements, instance_categories = get_instance_infoV3(all_disp_field_gt=all_disp_field_gt_numpy, \
                                                                                                valid_maps=all_valid_pixel_maps_numpy, \
                                                                                                pixel_cat_maps_gt=pixel_cat_map_gt_numpy, \
                                                                                                pixel_instance_maps_gt=pixel_instance_map_gt_numpy, \
                                                                                                num_grid=100, \
                                                                                                        )
        elif args.instance_mode == "grid":
            instance_positions, instance_displacements, instance_categories = get_instance_infoV2(all_disp_field_gt=all_disp_field_gt_numpy, \
                                                                                                valid_maps=all_valid_pixel_maps_numpy, \
                                                                                                pixel_cat_maps_gt=pixel_cat_map_gt_numpy, \
                                                                                                pixel_instance_maps_gt=pixel_instance_map_gt_numpy, \
                                                                                                num_grid=100, \
                                                                                                        )
        if args.grid_prior:
            # scene_gt = (pixel_cat_map_gt_clone, all_disp_field_gt_clone, motion_gt_clone)
            scene_gt = torch.cat([pixel_cat_map_gt_clone, all_disp_field_gt_clone, motion_gt_clone], dim=1)
            cell_gt = pixel_cat_map_gt_clone
        else:
            scene_gt = None
            # cell_gt = pixel_cat_map_gt_clone
        # breakpoint()
        instance_info = (instance_positions, instance_displacements, instance_categories )
        # disp_pred, class_pred, motion_pred, vae_dict = model(padded_voxel_points, scene_gt, instance_info=instance_info)
        
        disp_pred, class_pred, motion_pred, vae_dict, aug_disp_list = model(bevs=padded_voxel_points, disp_gt_map=scene_gt, instance_info=instance_info, \
                                                                dir_aug=args.use_dir_aug)  
                                                                
        

    
        optimizer.zero_grad()
        # compute reconstruct loss
        loss_all, loss_disp, loss_class, loss_motion, loss_bg_tc, loss_sc, loss_fg_tc = \
                compute_aux_loss(optimizer, device, num_future_frames[0].item(), all_disp_field_gt, all_valid_pixel_maps,
                            pixel_cat_map_gt, disp_pred, criterion, non_empty_map, class_pred, motion_gt,
                            motion_pred, trans_matrices, pixel_instance_map)

        # compute augment loss
        # loss_aug_list = []
        total_loss_aug = 0.0
        if args.use_dir_aug:
            for aug_disp in aug_disp_list:
                loss_aug, loss_aug_disp, loss_bg_tc, loss_sc, loss_fg_tc \
                    = compute_aug_loss(optimizer, device, num_future_frames[0].item(), all_disp_field_gt, all_valid_pixel_maps,
                                pixel_cat_map_gt, aug_disp, criterion, non_empty_map, class_pred, motion_gt,
                                motion_pred, trans_matrices, pixel_instance_map)
                total_loss_aug += loss_aug
            total_loss_aug = total_loss_aug / len(aug_disp_list)
            loss_all += 0.5 * total_loss_aug
        
        # compute vae kld loss
        vae_loss = compute_vae_loss(vae_dict)
        re_weight = 5.0
        kld_weight = 0.5
        loss = re_weight * loss_all + kld_weight * vae_loss
        # torch.autograd.set_detect_anomaly(True)
        # with torch.autograd.detect_anomaly():
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


        if not all((loss_disp, loss_class, loss_motion)):
            print("{}, \t{}, \tat epoch {}, \titerations {} [empty occupy map]".
                  format(running_aux_loss_disp, running_aux_loss_class, epoch, i))
            continue

        running_loss_bg_tc.update(loss_bg_tc)
        running_loss_fg_tc.update(loss_fg_tc)
        running_loss_sc.update(loss_sc)
        running_aux_loss_disp.update(loss_disp)
        running_aux_loss_class.update(loss_class)
        running_aux_loss_motion.update(loss_motion)
        running_loss_D.update(-1)
        running_loss_G.update(-1)
        running_loss_kld.update(vae_loss.item())
            
        if i % 10 == 0:
            print("[{}/{}]\t{}, \t{}, \t{}, \t{}, \t{}, \n\t{}, \t{} \t{}, \n\t{}".
                format(epoch, i, running_loss_D, running_loss_G, \
                       running_aux_loss_disp, running_aux_loss_class, running_aux_loss_motion,
                       running_loss_bg_tc,
                     running_loss_sc, running_loss_fg_tc, running_loss_kld))
    return running_loss_D, running_loss_G, running_loss_bg_tc, running_loss_sc, running_loss_fg_tc, \
           running_aux_loss_disp, running_aux_loss_class, running_aux_loss_motion, running_loss_kld

def eval(model, D, dataloader, dataset, device, use_adj_frame_pred=True,
        future_frame_skip=0, num_future_sweeps=20, batch_size=1, use_motion_state_pred_masking=True,
        general_flag = False):
    """Evaluate the motion prediction results for every epoch."""

    # The speed intervals for grouping the cells
    # speed intervals unit: m/s
    # We do not consider > 20m/s
    speed_intervals = np.array([ [0.0, 0.0], [0.0, 5.0], [5.0, 20.0] ])

    # evaluate predictions at [0.2, 0.4, ..., 1]s
    selected_future_sweeps = np.arange(0, num_future_sweeps + 1, future_frame_skip + 1)
    selected_future_sweeps = selected_future_sweeps[1:]
    last_future_sweep_id = selected_future_sweeps[-1]
    # lidar scanner is 20HZ
    distance_intervals = speed_intervals * (last_future_sweep_id / num_future_sweeps)

    # grouping the cells according to the speed intervals
 
    cell_groups = list()
    for i in range(distance_intervals.shape[0]):
        cell_statistics = list()

        for j in range(len(selected_future_sweeps)):
            # corresponds to each row, which records the MSE, median etc.
            cell_statistics.append([])
        cell_groups.append(cell_statistics)
    
     # Make prediction

    pixel_acc = 0  # for computing mean pixel classification accuracy
    overall_cls_pred = list()  # to compute classification accuracy for each object category
    overall_cls_gt = list()  # to compute classification accuracy for each object category
   
    for i, data in enumerate(dataloader, 0):
        
        
        if args.sensor_type == 'nuscenes' or args.sensor_type == 'at128_0':
            padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, \
            non_empty_map, pixel_cat_map_gt, pixel_instance_map_gt, past_steps, future_steps, motion_gt = data
        
        elif args.sensor_type == 'wt128_0':
            padded_voxel_points, all_disp_field_gt, all_valid_pixel_maps, \
                non_empty_map, pixel_cat_map_gt, pixel_instance_map_gt, past_steps, future_steps, motion_gt, \
            all_pixel_velocity_maps, all_pixel_direction_maps, all_pixel_cos_maps = data
            
            all_pixel_velocity_maps = all_pixel_velocity_maps.view(-1, 5, H, W, 1).to(device)
            all_pixel_direction_maps = all_pixel_direction_maps.view(-1, 5, H, W, 1).to(device)
            all_pixel_cos_maps = all_pixel_cos_maps.view(-1, 5, H, W, 3).to(device)
        
        padded_voxel_points = padded_voxel_points.to(device)

        if general_flag:
            # segment
            pixel_cat_map_gt_numpy = pixel_cat_map_gt.numpy()
            max_prob = np.amax(pixel_cat_map_gt_numpy, axis=-1)
            filter_mask = max_prob==1.0
            pixel_cat_map_numpy = np.argmax(pixel_cat_map_gt_numpy, axis=-1) + 1


            # convert category label to nongeneral label
            cat_nongeneral_map = np.zeros_like(pixel_cat_map_numpy)
            cat_nongeneral_map[pixel_cat_map_numpy == 5] = 1

        # noise sample
        # noise:torch.tensor = noise_sample_ACGAN(batch_size=padded_voxel_points.shape[0], n_z=32, n_size=256).to(device)
        noise:torch.tensor = noise_sample_ACGAN(batch_size=padded_voxel_points.shape[0], n_z=100, n_size=8).to(device)
        noise = noise * NOISE_FACTOR
        with torch.no_grad():
            # Make prediction, aux prediction
            # disp_pred, class_pred, motion_pred, _ = model(padded_voxel_points, )
            if "Uni" in args.model:
                if args.use_v:
                    if args.use_vdir:
                        disp_pred, class_pred, motion_pred, _, _ = model(padded_voxel_points, gt_map=None, \
                                prior_velocity=all_pixel_velocity_maps, prior_direction=all_pixel_direction_maps)
                        # disp_pred, class_pred, motion_pred = model(padded_voxel_points, all_pixel_velocity_maps, None)
                    else:
                        disp_pred, class_pred, motion_pred, _, _= model(padded_voxel_points, gt_map=None, \
                                prior_velocity=all_pixel_velocity_maps, prior_direction=None)
                else:
                    disp_pred, class_pred, motion_pred, _, _ = model(padded_voxel_points)
            else:
                if args.use_v:
                    if args.use_vdir:
                        disp_pred, class_pred, motion_pred, _, = model(padded_voxel_points, gt_map=None, \
                                prior_velocity=all_pixel_velocity_maps, prior_direction=all_pixel_direction_maps)
                        # disp_pred, class_pred, motion_pred = model(padded_voxel_points, all_pixel_velocity_maps, None)
                    else:
                        disp_pred, class_pred, motion_pred, _= model(padded_voxel_points, gt_map=None, \
                                prior_velocity=all_pixel_velocity_maps, prior_direction=None)
                else:
                    disp_pred, class_pred, motion_pred, _= model(padded_voxel_points)
            # (bs*frames, 2, h, w)
            pred_shape = disp_pred.size()

            # (bs, frames, 2, h, w)
            disp_pred = disp_pred.view(all_disp_field_gt.size(0), -1, pred_shape[-3], pred_shape[-2], pred_shape[-1])
            disp_pred = disp_pred.contiguous()
            disp_pred = disp_pred.cpu().numpy()
            
            if use_adj_frame_pred:
                for c in range(1, disp_pred.shape[1]):
                    disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]

            if use_motion_state_pred_masking:
                # static mask
                motion_pred_numpy = motion_pred.cpu().numpy()
                motion_pred_numpy = np.argmax(motion_pred_numpy, axis=1)
                mask = motion_pred_numpy == 0

                class_pred_numpy = class_pred.cpu().numpy()
                class_pred_cat = np.argmax(class_pred_numpy, axis=1)

                # background mask
                class_mask = class_pred_cat == 0

                # For those with very small movements, we consider them as static
                last_pred = disp_pred[:, -1, ...]
                # (bs, h, w)
                last_pred_norm = np.linalg.norm(last_pred, ord=2, axis=1)
                thd_mask = last_pred_norm <= 0.2

                cat_weight_map = np.ones_like(class_pred_cat, dtype=np.float32)
                cat_weight_map[mask] = 0.0
                cat_weight_map[class_mask] = 0.0
                cat_weight_map[thd_mask] = 0.0
                # (bs, 1, 1, h, w)
                cat_weight_map = cat_weight_map[:, np.newaxis, np.newaxis, ...]


                disp_pred = disp_pred * cat_weight_map

                if general_flag:
                    disp_pred = disp_pred * cat_nongeneral_map

        # Pre-processing for gt
        # (bs, seq, h, w, channel=2)
        all_disp_field_gt = all_disp_field_gt.numpy()
        future_steps = future_steps.numpy()[0]

        valid_pixel_maps = all_valid_pixel_maps[:, -future_steps:, ...].contiguous()
        valid_pixel_maps = valid_pixel_maps.numpy()

        all_disp_field_gt = all_disp_field_gt[:, -future_steps:, ...]
        all_disp_field_gt = np.transpose(all_disp_field_gt, (0,1,4,2,3))
        all_disp_field_gt_norm = np.linalg.norm(all_disp_field_gt, ord=2, axis=2)

        """compute the evaluation metrics"""
        upper_thresh = 0.2
        upper_bound = (future_frame_skip + 1)/num_future_sweeps * upper_thresh
        static_cell_mask = all_disp_field_gt_norm <= upper_bound
        static_cell_mask = np.all(static_cell_mask, axis=1)
        moving_cell_mask = np.logical_not(static_cell_mask)

        for j, d in enumerate(distance_intervals):
            for slot, s in enumerate((selected_future_sweeps - 1)):  # selected_future_sweeps: [4, 8, ...]
                curr_valid_pixel_map = valid_pixel_maps[:, s]

                if j == 0:  # corresponds to static cells
                    curr_mask = np.logical_and(curr_valid_pixel_map, static_cell_mask)
                else:
                    # We use the displacement between keyframe and the last sample frame as metrics
                    last_gt_norm = all_disp_field_gt_norm[:, -1]
                    mask = np.logical_and(d[0] <= last_gt_norm, last_gt_norm < d[1])

                    curr_mask = np.logical_and(curr_valid_pixel_map, mask)
                    curr_mask = np.logical_and(curr_mask, moving_cell_mask)

                # Since in nuScenes (with 32-line LiDAR) the points (cells) in the distance are very sparse,
                # we evaluate the performance for cells within the range [-30m, 30m] along both x, y dimensions.
                border = 8
                roi_mask = np.zeros_like(curr_mask, dtype=bool)
                roi_mask[:, border:-border, border:-border] = True
                curr_mask = np.logical_and(curr_mask, roi_mask)

                cell_idx = np.where(curr_mask == True)

                if general_flag:
                    all_disp_field_gt = all_disp_field_gt * cat_nongeneral_map

                gt = all_disp_field_gt[:, s]
                pred = disp_pred[:, s]
                norm_error = np.linalg.norm(gt - pred, ord=2, axis=1)

                cell_groups[j][slot].append(norm_error[cell_idx])

        # -----------------------------------------------------------------------------------
        # Second, compute the classification accuracy
        pixel_cat_map_gt_numpy = pixel_cat_map_gt.numpy()
        non_empty_map_numpy = non_empty_map.numpy()
        class_pred_numpy = class_pred.cpu().numpy()

        # Convert the category map
        max_prob = np.amax(pixel_cat_map_gt_numpy, axis=-1)
        filter_mask = max_prob == 1.0  # Note: some of the cell probabilities are soft probabilities
        pixel_cat_map_numpy = np.argmax(pixel_cat_map_gt_numpy, axis=-1) + 1  # category starts from 1 (background), etc
        pixel_cat_map_numpy = (pixel_cat_map_numpy * non_empty_map_numpy * filter_mask).astype(int)

        class_pred_numpy = np.transpose(class_pred_numpy, (0, 2, 3, 1))
        class_pred_numpy = np.argmax(class_pred_numpy, axis=-1) + 1
        class_pred_numpy = (class_pred_numpy * non_empty_map_numpy * filter_mask).astype(int)

        border = 8
        roi_mask = np.zeros_like(non_empty_map_numpy)
        roi_mask[:, border:-border, border:-border] = 1.0

        tmp = pixel_cat_map_numpy == class_pred_numpy
        denominator = np.sum(non_empty_map_numpy * filter_mask * roi_mask)
        pixel_acc += np.sum(tmp * non_empty_map_numpy * filter_mask * roi_mask) / denominator

        # For computing confusion matrix, in order to compute classification accuracy for each category
        count_mask = non_empty_map_numpy * filter_mask * roi_mask
        idx_fg = np.where(count_mask > 0)

        overall_cls_gt.append(pixel_cat_map_numpy[idx_fg])
        overall_cls_pred.append(class_pred_numpy[idx_fg])
        if (i + 1) % 100 == 0:

            print("Finish sample [{}/{}]".format(i + 1, int(np.ceil(len(dataset) / float(batch_size)))))

    # Compute the statistics
    dump_res = []

    # Compute the statistics of displacement prediction error
    me_list = np.zeros([3])
    quantile_50 = np.zeros([3])
    for i, d in enumerate(speed_intervals):
        group = cell_groups[i]
        print("--------------------------------------------------------------")
        print("For cells within speed range [{}, {}]:\n".format(d[0], d[1]))
        # if save_eval_file_path is not None:
        #     saver.write("--------------------------------------------------------------\n")
        #     saver.write("For cells within speed range [{}, {}]:\n\n".format(d[0], d[1]))

        dump_error = []
        dump_error_quantile_50 = []

        for s in range(len(selected_future_sweeps)):
            row = group[s]

            errors = np.concatenate(row) if len(row) != 0 else row

            if len(errors) == 0:
                mean_error = None
                error_quantile_50 = None
            else:
                mean_error = np.average(errors)
                error_quantile_50 = np.quantile(errors, 0.5)

            dump_error.append(mean_error)
            dump_error_quantile_50.append(error_quantile_50)

            msg = "Frame {}:\nThe mean error is {}\nThe 50% error quantile is {}".\
                format(selected_future_sweeps[s], mean_error, error_quantile_50)
            print(msg)
        me_list[i] = mean_error
        quantile_50[i] = error_quantile_50
        print("--------------------------------------------------------------\n")
        # if save_eval_file_path is not None:
        #     saver.write("--------------------------------------------------------------\n\n")

        dump_res.append(dump_error + dump_error_quantile_50)

    # Compute the statistics of mean pixel classification accuracy
    pixel_acc = pixel_acc / len(dataset)
    print("Mean pixel classification accuracy: {}".format(pixel_acc))
    # if save_eval_file_path is not None:
    #     saver.write("Mean pixel classification accuracy: {}\n".format(pixel_acc))

    # Compute the mean classification accuracy for each object category
    overall_cls_gt = np.concatenate(overall_cls_gt)
    overall_cls_pred = np.concatenate(overall_cls_pred)
    cm = confusion_matrix(overall_cls_gt, overall_cls_pred)
    cm_sum = np.sum(cm, axis=1)
    cls_num = cm.shape[0]
    mean_cat = cm[np.arange(cls_num), np.arange(cls_num)] / cm_sum
    cat_map = {0: 'Bg', 1: 'Vehicle', 2: 'Ped', 3: 'Bike', 4: 'Others'}
    for i in range(len(mean_cat)):
        print("mean cat accuracy of {}: {}".format(cat_map[i], mean_cat[i]))
    print("mean instance acc: ", np.mean(mean_cat))
  
    return me_list[0], me_list[1], me_list[2], quantile_50[0], quantile_50[1], quantile_50[2], \
            cat_map, mean_cat, pixel_acc

def compute_vae_loss(output):
    """compute vae loss"""

    present_mu = output['present_mu']
    present_log_sigma = output['present_log_sigma']
    future_mu = output['future_mu']
    future_log_sigma = output['future_log_sigma']

    var_future = torch.exp(2 * future_log_sigma)
    var_present = torch.exp(2 * present_log_sigma)
    kl_div = (
            present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                2 * var_present )
    )
    # breakpoint()
    # print(var_future, var_present)
    kl_loss = torch.mean(torch.sum(kl_div, dim=-1))

    return kl_loss

def compute_aux_loss(optimizer, device, future_frames_num, all_disp_field_gt, all_valid_pixel_maps, pixel_cat_map_gt,
                        disp_pred, criterion, non_empty_map, class_pred, motion_gt, motion_pred, trans_matrices,
                        pixel_instance_map):
    """compute aux loss"""
    
    # compute the displacement loss

    # (bs*2clips, step, h, w, 2)
    all_disp_field_gt = all_disp_field_gt.view(-1, future_frames_num, H, W, 2)
    gt = all_disp_field_gt[:, -future_frames_num:, ...].contiguous()
    gt = gt.view(-1, gt.size(2), gt.size(3), gt.size(4))
    gt = gt.permute(0, 3, 1, 2).to(device)

    all_valid_pixel_maps = all_valid_pixel_maps.view(-1, future_frames_num, H, W)
    valid_pixel_maps = all_valid_pixel_maps[:, -future_frames_num:, ...].contiguous()
    valid_pixel_maps = valid_pixel_maps.view(-1, valid_pixel_maps.size(2), valid_pixel_maps.size(3))
    valid_pixel_maps = torch.unsqueeze(valid_pixel_maps, 1)
    valid_pixel_maps = valid_pixel_maps.to(device)

    valid_pixel_num = torch.nonzero(valid_pixel_maps).size(0)
    if valid_pixel_num == 0:
        return [None] * 6

    # generate the displacement w.r.t. the keyframe
    if pred_adj_frame_distance:
        disp_pred = disp_pred.view(-1, future_frames_num, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))

        # compute temporal consistency loss
        if use_bg_temporal_consistency:
            bg_tc_loss = background_temporal_consistency_loss(disp_pred, pixel_cat_map_gt, non_empty_map, trans_matrices)

        if use_fg_temporal_consistency or use_spatial_consistency:
            instance_spatio_temp_loss, instance_spatial_loss_value, instance_temporal_loss_value = \
                instance_spatial_temporal_consistency_loss(disp_pred, pixel_instance_map)

        for c in range(1, disp_pred.size(1)):
            disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]
        disp_pred = disp_pred.view(-1, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))
        
    """compute the masked displacement loss"""
    pixel_cat_map_gt = pixel_cat_map_gt.view(-1, H, W, cell_category_num)

    if use_weighted_loss:
        pixel_cat_map_gt_numpy = pixel_cat_map_gt.cpu().data.numpy()
        pixel_cat_map_gt_numpy = np.argmax(pixel_cat_map_gt_numpy, axis=-1) + 1

        cat_weight_map = np.zeros_like(pixel_cat_map_gt_numpy, dtype=np.float32)
        cat_weight_map_cls = cat_weight_map
        # [bg, vehicle, pedestrian, bike, other]
        weight_vector = [0.005, 1.0, 1.0, 1.0, 1.0]

        if get_general:
            curr_annotated_point_num = np.sum( (pixel_cat_map_gt_numpy != 5) & (pixel_cat_map_gt_numpy != 0) )
            valid_pixel_num = curr_annotated_point_num
            weight_vector_general = [0.005, 1.0, 1.0, 1.0, 0.0]

        for k in range(len(weight_vector)):
            mask = pixel_cat_map_gt_numpy == (k + 1)
            if get_general:
               cat_weight_map[mask] = weight_vector_general[k]
               cat_weight_map_cls[mask] = weight_vector[k]
            
            else:
                cat_weight_map[mask] = weight_vector[k]
                cat_weight_map_cls[mask] = weight_vector[k]

        # shape: [bs, 1, 1, h, w]
        cat_weight_map = cat_weight_map[:, np.newaxis, np.newaxis, ...]  # (batch, 1, 1, h, w)
        cat_weight_map = torch.from_numpy(cat_weight_map).to(device)
        cat_weight_map_cls = cat_weight_map_cls[:, np.newaxis, np.newaxis, ...]
        cat_weight_map_cls = torch.from_numpy(cat_weight_map_cls).to(device)
        map_shape = cat_weight_map.size()

        # loss disp size (bs*2clips*seq, 2, h ,w)
        loss_disp = criterion(gt * valid_pixel_maps, disp_pred * valid_pixel_maps)
        loss_disp = loss_disp.reshape(map_shape[0], -1, map_shape[-3], map_shape[-2], map_shape[-1])

        if use_disp_weighted_loss:
            disp_weight_map = choose_disp_weighted_map(gt=all_disp_field_gt, map_shape=map_shape, device=device, mode=args.dw_mode)
            loss_disp = torch.sum(loss_disp * cat_weight_map * disp_weight_map) / valid_pixel_num
        else:
            loss_disp = torch.sum(loss_disp * cat_weight_map) / valid_pixel_num
    else:
        loss_disp = criterion(gt * valid_pixel_maps, disp_pred * valid_pixel_maps) / valid_pixel_num

    # compute the grid cell classification loss
    """compute the grid cell classification loss"""
    cat_weight_map = cat_weight_map_cls

    non_empty_map = non_empty_map.view(-1, H, W)
    non_empty_map = non_empty_map.to(device)
    # (bs, 5, h, w)
    pixel_cat_map_gt = pixel_cat_map_gt.permute(0, 3, 1, 2).to(device)
    
    log_softmax_probs = F.log_softmax(class_pred, dim=1)

    if use_weighted_loss:
        map_shape = cat_weight_map.size()
        cat_weight_map = cat_weight_map.view(map_shape[0], map_shape[-2], map_shape[-1]) # (batch, h, w)

        loss_class = torch.sum(- pixel_cat_map_gt * log_softmax_probs, dim=1) * cat_weight_map
    else:
        loss_class = torch.sum(- pixel_cat_map_gt * log_softmax_probs, dim=1)
    loss_class = torch.sum(loss_class * non_empty_map) / torch.nonzero(non_empty_map).size(0)

    """compute the motion loss"""
    motion_gt = motion_gt.view(-1, H, W, 2)
    motion_gt_numpy = motion_gt.numpy()
    motion_gt = motion_gt.permute(0, 3, 1, 2).to(device)
    log_softmax_motion_pred = F.log_softmax(motion_pred, dim=1)

    if use_weighted_loss:
        motion_gt_numpy = np.argmax(motion_gt_numpy, axis=-1) + 1
        motion_weight_map = np.zeros_like(motion_gt_numpy, dtype=np.float32)
        weight_vector = [0.005, 1.0] # [static, motion]
        for k in range(len(weight_vector)):
            mask = motion_gt_numpy == (k+1)
            motion_weight_map[mask] = weight_vector[k]
        
        motion_weight_map = torch.from_numpy(motion_weight_map).to(device)
        loss_speed = torch.sum(- motion_gt * log_softmax_motion_pred, dim=1) * motion_weight_map
    else:
        loss_speed = torch.sum(- motion_gt * log_softmax_motion_pred, dim=1)    
    loss_motion = torch.sum(loss_speed * non_empty_map) / torch.nonzero(non_empty_map).size(0)


    """sum up all losses"""
    if use_bg_temporal_consistency and (use_fg_temporal_consistency or use_spatial_consistency):
        loss = loss_disp + loss_class + loss_motion + reg_weight_bg_tc * bg_tc_loss + instance_spatio_temp_loss
    elif use_bg_temporal_consistency:
        loss = loss_disp + loss_class + loss_motion + reg_weight_bg_tc * bg_tc_loss
    elif use_spatial_consistency or use_fg_temporal_consistency:
        loss = loss_disp + loss_class + loss_motion + instance_spatio_temp_loss
    else:
        loss = loss_disp + loss_class + loss_motion
    
    loss_aux = loss

    if use_bg_temporal_consistency:
        loss_bg_tc_value = bg_tc_loss.item()
    else:
        loss_bg_tc_value = -1

    if use_spatial_consistency or use_fg_temporal_consistency:
        sc_loss_value = instance_spatial_loss_value
        fg_tc_loss_value = instance_temporal_loss_value
    else:
        sc_loss_value = -1
        fg_tc_loss_value = -1
    
    return loss_aux, loss_disp.item(), loss_class.item(), loss_motion.item(), loss_bg_tc_value, sc_loss_value, fg_tc_loss_value

def compute_aug_loss(optimizer, device, future_frames_num, all_disp_field_gt, all_valid_pixel_maps, pixel_cat_map_gt,
                        disp_pred, criterion, non_empty_map, class_pred, motion_gt, motion_pred, trans_matrices,
                        pixel_instance_map):
    """compute aux loss"""
    
    # compute the displacement loss

    # (bs*2clips, step, h, w, 2)
    all_disp_field_gt = all_disp_field_gt.view(-1, future_frames_num, H, W, 2)
    gt = all_disp_field_gt[:, -future_frames_num:, ...].contiguous()
    gt = gt.view(-1, gt.size(2), gt.size(3), gt.size(4))
    gt = gt.permute(0, 3, 1, 2).to(device)

    all_valid_pixel_maps = all_valid_pixel_maps.view(-1, future_frames_num, H, W)
    valid_pixel_maps = all_valid_pixel_maps[:, -future_frames_num:, ...].contiguous()
    valid_pixel_maps = valid_pixel_maps.view(-1, valid_pixel_maps.size(2), valid_pixel_maps.size(3))
    valid_pixel_maps = torch.unsqueeze(valid_pixel_maps, 1)
    valid_pixel_maps = valid_pixel_maps.to(device)

    valid_pixel_num = torch.nonzero(valid_pixel_maps).size(0)
    if valid_pixel_num == 0:
        return [None] * 6

    # generate the displacement w.r.t. the keyframe
    if pred_adj_frame_distance:
        disp_pred = disp_pred.view(-1, future_frames_num, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))

        # compute temporal consistency loss
        if use_bg_temporal_consistency:
            bg_tc_loss = background_temporal_consistency_loss(disp_pred, pixel_cat_map_gt, non_empty_map, trans_matrices)

        if use_fg_temporal_consistency or use_spatial_consistency:
            instance_spatio_temp_loss, instance_spatial_loss_value, instance_temporal_loss_value = \
                instance_spatial_temporal_consistency_loss(disp_pred, pixel_instance_map)

        for c in range(1, disp_pred.size(1)):
            disp_pred[:, c, ...] = disp_pred[:, c, ...] + disp_pred[:, c - 1, ...]
        disp_pred = disp_pred.view(-1, disp_pred.size(-3), disp_pred.size(-2), disp_pred.size(-1))
        
    """compute the masked displacement loss"""
    pixel_cat_map_gt = pixel_cat_map_gt.view(-1, H, W, cell_category_num)

    if use_weighted_loss:
        pixel_cat_map_gt_numpy = pixel_cat_map_gt.cpu().data.numpy()
        pixel_cat_map_gt_numpy = np.argmax(pixel_cat_map_gt_numpy, axis=-1) + 1

        cat_weight_map = np.zeros_like(pixel_cat_map_gt_numpy, dtype=np.float32)
        cat_weight_map_cls = cat_weight_map
        # [bg, vehicle, pedestrian, bike, other]
        weight_vector = [0.005, 1.0, 1.0, 1.0, 1.0]

        if get_general:
            curr_annotated_point_num = np.sum( (pixel_cat_map_gt_numpy != 5) & (pixel_cat_map_gt_numpy != 0) )
            valid_pixel_num = curr_annotated_point_num
            weight_vector_general = [0.005, 1.0, 1.0, 1.0, 0.0]

        for k in range(len(weight_vector)):
            mask = pixel_cat_map_gt_numpy == (k + 1)
            if get_general:
               cat_weight_map[mask] = weight_vector_general[k]
               cat_weight_map_cls[mask] = weight_vector[k]
            
            else:
                cat_weight_map[mask] = weight_vector[k]
                cat_weight_map_cls[mask] = weight_vector[k]

        # shape: [bs, 1, 1, h, w]
        cat_weight_map = cat_weight_map[:, np.newaxis, np.newaxis, ...]  # (batch, 1, 1, h, w)
        cat_weight_map = torch.from_numpy(cat_weight_map).to(device)
        cat_weight_map_cls = cat_weight_map_cls[:, np.newaxis, np.newaxis, ...]
        cat_weight_map_cls = torch.from_numpy(cat_weight_map_cls).to(device)
        map_shape = cat_weight_map.size()

        # loss disp size (bs*2clips*seq, 2, h ,w)
        loss_disp = criterion(gt * valid_pixel_maps, disp_pred * valid_pixel_maps)
        loss_disp = loss_disp.reshape(map_shape[0], -1, map_shape[-3], map_shape[-2], map_shape[-1])

        if use_disp_weighted_loss:
            disp_weight_map = choose_disp_weighted_map(gt=all_disp_field_gt, map_shape=map_shape, device=device, mode=args.dw_mode)
            loss_disp = torch.sum(loss_disp * cat_weight_map * disp_weight_map) / valid_pixel_num
        else:
            loss_disp = torch.sum(loss_disp * cat_weight_map) / valid_pixel_num
    else:
        loss_disp = criterion(gt * valid_pixel_maps, disp_pred * valid_pixel_maps) / valid_pixel_num



    """sum up all losses"""
    if use_bg_temporal_consistency and (use_fg_temporal_consistency or use_spatial_consistency):
        loss = loss_disp + reg_weight_bg_tc * bg_tc_loss + instance_spatio_temp_loss
    elif use_bg_temporal_consistency:
        loss = loss_disp + reg_weight_bg_tc * bg_tc_loss
    elif use_spatial_consistency or use_fg_temporal_consistency:
        loss = loss_disp + instance_spatio_temp_loss
    else:
        loss = loss_disp
    
    loss_aux = loss

    if use_bg_temporal_consistency:
        loss_bg_tc_value = bg_tc_loss.item()
    else:
        loss_bg_tc_value = -1

    if use_spatial_consistency or use_fg_temporal_consistency:
        sc_loss_value = instance_spatial_loss_value
        fg_tc_loss_value = instance_temporal_loss_value
    else:
        sc_loss_value = -1
        fg_tc_loss_value = -1
    
    return loss_aux, loss_disp.item(), loss_bg_tc_value, sc_loss_value, fg_tc_loss_value

def background_temporal_consistency_loss(disp_pred, pixel_cat_map_gt, non_empty_map, trans_matrices):
    """
    disp_pred: Should be relative displacement between adjacent frames. shape (batch * 2, sweep_num, 2, h, w)
    pixel_cat_map_gt: Shape (batch, 2, h, w, cat_num)
    non_empty_map: Shape (batch, 2, h, w)
    trans_matrices: Shape (batch, 2, sweep_num, 4, 4)
    """
    criterion = nn.SmoothL1Loss(reduction='sum')

    non_empty_map_numpy = non_empty_map.numpy()
    pixel_cat_maps = pixel_cat_map_gt.numpy()
    max_prob = np.amax(pixel_cat_maps, axis=-1)
    filter_mask = max_prob == 1.0
    pixel_cat_maps = np.argmax(pixel_cat_maps, axis=-1) + 1  # category starts from 1 (background), etc
    pixel_cat_maps = (pixel_cat_maps * non_empty_map_numpy * filter_mask)  # (batch, 2, h, w)

    trans_matrices = trans_matrices.numpy()
    device = disp_pred.device

    pred_shape = disp_pred.size()
    disp_pred = disp_pred.view(-1, 2, pred_shape[1], pred_shape[2], pred_shape[3], pred_shape[4])

    seq_1_pred = disp_pred[:, 0]  # (batch, sweep_num, 2, h, w)
    seq_2_pred = disp_pred[:, 1]

    seq_1_absolute_pred_list = list()
    seq_2_absolute_pred_list = list()

    seq_1_absolute_pred_list.append(seq_1_pred[:, 1])
    for i in range(2, pred_shape[1]):
        seq_1_absolute_pred_list.append(seq_1_pred[:, i] + seq_1_absolute_pred_list[i - 2])

    seq_2_absolute_pred_list.append(seq_2_pred[:, 0])
    for i in range(1, pred_shape[1] - 1):
        seq_2_absolute_pred_list.append(seq_2_pred[:, i] + seq_2_absolute_pred_list[i - 1])

    # ----------------- Compute the consistency loss -----------------
    # Compute the transformation matrices
    # First, transform the coordinate
    transformed_disp_pred_list = list()

    trans_matrix_global = trans_matrices[:, 1]  # (batch, sweep_num, 4, 4)
    trans_matrix_global = trans_matrix_global[:, trans_matrix_idx, 0:3]  # (batch, 3, 4)  # <---
    trans_matrix_global = trans_matrix_global[:, :, (0, 1, 3)]  # (batch, 3, 3)
    trans_matrix_global[:, 2] = np.array([0.0, 0.0, 1.0])

    # --- Move pixel coord to global and rescale; then rotate; then move back to local pixel coord
    translate_to_global = np.array([[1.0, 0.0, -120.0], [0.0, 1.0, -120.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    scale_global = np.array([[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    trans_global = scale_global @ translate_to_global
    inv_trans_global = np.linalg.inv(trans_global)

    trans_global = np.expand_dims(trans_global, axis=0)
    inv_trans_global = np.expand_dims(inv_trans_global, axis=0)
    trans_matrix_total = inv_trans_global @ trans_matrix_global @ trans_global

    # --- Generate grid transformation matrix, so as to use Pytorch affine_grid and grid_sample function
    w, h = pred_shape[-2], pred_shape[-1]
    resize_m = np.array([
        [2 / w, 0.0, -1],
        [0.0, 2 / h, -1],
        [0.0, 0.0, 1]
    ], dtype=np.float32)
    inverse_m = np.linalg.inv(resize_m)
    resize_m = np.expand_dims(resize_m, axis=0)
    inverse_m = np.expand_dims(inverse_m, axis=0)

    grid_trans_matrix = resize_m @ trans_matrix_total @ inverse_m  # (batch, 3, 3)
    grid_trans_matrix = grid_trans_matrix[:, :2].astype(np.float32)
    grid_trans_matrix = torch.from_numpy(grid_trans_matrix)

    # --- For displacement field
    trans_matrix_translation_global = np.eye(trans_matrix_total.shape[1])
    trans_matrix_translation_global = np.expand_dims(trans_matrix_translation_global, axis=0)
    trans_matrix_translation_global = np.repeat(trans_matrix_translation_global, grid_trans_matrix.shape[0], axis=0)
    trans_matrix_translation_global[:, :, 2] = trans_matrix_global[:, :, 2]  # only translation
    trans_matrix_translation_total = inv_trans_global @ trans_matrix_translation_global @ trans_global

    grid_trans_matrix_disp = resize_m @ trans_matrix_translation_total @ inverse_m
    grid_trans_matrix_disp = grid_trans_matrix_disp[:, :2].astype(np.float32)
    grid_trans_matrix_disp = torch.from_numpy(grid_trans_matrix_disp).to(device)

    disp_rotate_matrix = trans_matrix_global[:, 0:2, 0:2].astype(np.float32)  # (batch, 2, 2)
    disp_rotate_matrix = torch.from_numpy(disp_rotate_matrix).to(device)

    for i in range(len(seq_1_absolute_pred_list)):

        # --- Start transformation for displacement field
        curr_pred = seq_1_absolute_pred_list[i]  # (batch, 2, h, w)

        # First, rotation
        curr_pred = curr_pred.permute(0, 2, 3, 1).contiguous()  # (batch, h, w, 2)
        curr_pred = curr_pred.view(-1, h * w, 2)
        curr_pred = torch.bmm(curr_pred, disp_rotate_matrix)
        curr_pred = curr_pred.view(-1, h, w, 2)
        curr_pred = curr_pred.permute(0, 3, 1, 2).contiguous()  # (batch, 2, h, w)

        # Next, translation
        curr_pred = curr_pred.permute(0, 1, 3, 2).contiguous()  # swap x and y axis
        curr_pred = torch.flip(curr_pred, dims=[2])

        grid = F.affine_grid(grid_trans_matrix_disp, curr_pred.size())
        if use_nn_sampling:
            curr_pred = F.grid_sample(curr_pred, grid, mode='nearest')
        else:
            curr_pred = F.grid_sample(curr_pred, grid)

        curr_pred = torch.flip(curr_pred, dims=[2])
        curr_pred = curr_pred.permute(0, 1, 3, 2).contiguous()

        transformed_disp_pred_list.append(curr_pred)

    # --- Start transformation for category map
    pixel_cat_map = pixel_cat_maps[:, 0]  # (batch, h, w)
    pixel_cat_map = torch.from_numpy(pixel_cat_map.astype(np.float32))
    pixel_cat_map = pixel_cat_map[:, None, :, :]  # (batch, 1, h, w)
    trans_pixel_cat_map = pixel_cat_map.permute(0, 1, 3, 2)  # (batch, 1, h, w), swap x and y axis
    trans_pixel_cat_map = torch.flip(trans_pixel_cat_map, dims=[2])

    grid = F.affine_grid(grid_trans_matrix, pixel_cat_map.size())
    trans_pixel_cat_map = F.grid_sample(trans_pixel_cat_map, grid, mode='nearest')

    trans_pixel_cat_map = torch.flip(trans_pixel_cat_map, dims=[2])
    trans_pixel_cat_map = trans_pixel_cat_map.permute(0, 1, 3, 2)

    # --- Compute the loss, using smooth l1 loss
    adj_pixel_cat_map = pixel_cat_maps[:, 1]
    adj_pixel_cat_map = torch.from_numpy(adj_pixel_cat_map.astype(np.float32))
    adj_pixel_cat_map = torch.unsqueeze(adj_pixel_cat_map, dim=1)

    mask_common = trans_pixel_cat_map == adj_pixel_cat_map
    mask_common = mask_common.float()
    non_empty_map_gpu = non_empty_map.to(device)
    non_empty_map_gpu = non_empty_map_gpu[:, 1:2, :, :]  # select the second sequence, keep dim
    mask_common = mask_common.to(device)
    mask_common = mask_common * non_empty_map_gpu

    loss_list = list()
    for i in range(len(seq_1_absolute_pred_list)):
        trans_seq_1_pred = transformed_disp_pred_list[i]  # (batch, 2, h, w)
        seq_2_pred = seq_2_absolute_pred_list[i]  # (batch, 2, h, w)

        trans_seq_1_pred = trans_seq_1_pred * mask_common
        seq_2_pred = seq_2_pred * mask_common

        num_non_empty_cells = torch.nonzero(mask_common).size(0)
        if num_non_empty_cells != 0:
            loss = criterion(trans_seq_1_pred, seq_2_pred) / num_non_empty_cells
            loss_list.append(loss)

    res_loss = torch.mean(torch.stack(loss_list, 0))

    return res_loss


# We name it instance spatial-temporal consistency loss because it involves each instance
def instance_spatial_temporal_consistency_loss(disp_pred, pixel_instance_map):
    device = disp_pred.device

    pred_shape = disp_pred.size()
    disp_pred = disp_pred.view(-1, 2, pred_shape[1], pred_shape[2], pred_shape[3], pred_shape[4])

    seq_1_pred = disp_pred[:, 0]  # (batch, sweep_num, 2, h, w)
    seq_2_pred = disp_pred[:, 1]

    pixel_instance_map = pixel_instance_map.numpy()
    batch = pixel_instance_map.shape[0]

    spatial_loss = 0.0
    temporal_loss = 0.0
    counter = 0
    criterion = nn.SmoothL1Loss()

    for i in range(batch):
        curr_batch_instance_maps = pixel_instance_map[i]
        seq_1_instance_map = curr_batch_instance_maps[0]
        seq_2_instance_map = curr_batch_instance_maps[1]

        seq_1_instance_ids = np.unique(seq_1_instance_map)
        seq_2_instance_ids = np.unique(seq_2_instance_map)

        common_instance_ids = np.intersect1d(seq_1_instance_ids, seq_2_instance_ids, assume_unique=True)

        seq_1_batch_pred = seq_1_pred[i]  # (sweep_num, 2, h, w)
        seq_2_batch_pred = seq_2_pred[i]

        for h in common_instance_ids:
            if h == 0:  # do not consider the background instance
                continue

            seq_1_mask = np.where(seq_1_instance_map == h)
            seq_1_idx_x = torch.from_numpy(seq_1_mask[0]).to(device)
            seq_1_idx_y = torch.from_numpy(seq_1_mask[1]).to(device)
            seq_1_selected_cells = seq_1_batch_pred[:, :, seq_1_idx_x, seq_1_idx_y]

            seq_2_mask = np.where(seq_2_instance_map == h)
            seq_2_idx_x = torch.from_numpy(seq_2_mask[0]).to(device)
            seq_2_idx_y = torch.from_numpy(seq_2_mask[1]).to(device)
            seq_2_selected_cells = seq_2_batch_pred[:, :, seq_2_idx_x, seq_2_idx_y]

            seq_1_selected_cell_num = seq_1_selected_cells.size(2)
            seq_2_selected_cell_num = seq_2_selected_cells.size(2)

            # for spatial loss
            if use_spatial_consistency:
                tmp_seq_1 = 0
                if seq_1_selected_cell_num > 1:
                    tmp_seq_1 = criterion(seq_1_selected_cells[:, :, :-1], seq_1_selected_cells[:, :, 1:])

                tmp_seq_2 = 0
                if seq_2_selected_cell_num > 1:
                    tmp_seq_2 = criterion(seq_2_selected_cells[:, :, :-1], seq_2_selected_cells[:, :, 1:])

                spatial_loss += tmp_seq_1 + tmp_seq_2

            if use_fg_temporal_consistency:
                seq_1_mean = torch.mean(seq_1_selected_cells, dim=2)
                seq_2_mean = torch.mean(seq_2_selected_cells, dim=2)
                temporal_loss += criterion(seq_1_mean, seq_2_mean)

            counter += 1

    if counter != 0:
        spatial_loss = spatial_loss / counter
        temporal_loss = temporal_loss / counter

    total_loss = reg_weight_sc * spatial_loss + reg_weight_fg_tc * temporal_loss

    spatial_loss_value = 0 if type(spatial_loss) == float else spatial_loss.item()
    temporal_loss_value = 0 if type(temporal_loss) == float else temporal_loss.item()

    return total_loss, spatial_loss_value, temporal_loss_value

if __name__ == "__main__":
    main()