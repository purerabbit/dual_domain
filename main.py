# System / Python
import os
import argparse
import logging
import random
import shutil
import time
import numpy as np
from tqdm import tqdm
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler#并行网路
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
# Custom
from Networks.network import ParallelNetwork as Network#导入网络
# from IXI_dataset import IXIData as Dataset
from mri_tools import rA, rAtA, rfft2
from data.utils import *
# from preprocessing import *
from mask.gen_mask import *
#导入fastmri数据集文件 导入方式可能有问题
# from .data.dataset import FASTMRIDataset as Dataset#引用包问题
from PIL import Image
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image
# from utils import *
from data.dataset import *
from loss import cal_loss

results_save_path='./save_images'

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#单GPU进行训练

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='self-supervised MRI reconstruction', help='name of experiment')
# parameters related to distributed training
parser.add_argument('--init-method', default='tcp://localhost:1836', help='initialization method')
parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--gpus', type=int, default=1, help='number of gpus per node')#放到一块GPU上进行训练
parser.add_argument('--world-size', type=int, default=None, help='world_size = nodes * gpus')
# parameters related to model
parser.add_argument('--use-init-weights', '-uit', type=bool, default=True, help='whether initialize model weights with defined types')
parser.add_argument('--init-type', type=str, default='xavier', help='type of initialize model weights')
parser.add_argument('--gain', type=float, default=1.0, help='gain in the initialization of model weights')
parser.add_argument('--num-layers', type=int, default=9, help='number of iterations')
# learning rate, batch size, and etc
parser.add_argument('--seed', type=int, default=30, help='random seed number')
parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--batch-size', type=int, default=1, help='batch size of single gpu')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--warmup-epochs', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--num-epochs', type=int, default=500, help='maximum number of epochs')
# parameters related to data and masks
parser.add_argument('--train-path', type=str, default='/data/lc/3Dunet/fastmri_train_3c.npz', help='path of training data')
parser.add_argument('--val-path', type=str, default='/data/lc/3Dunet/fastmri_val_3c.npz', help='path of validation data')
parser.add_argument('--test-path', type=str, default='/data/lc/3Dunet/fastmri_test_3c.npz', help='path of test data')
#改变了数据集路径
parser.add_argument('--u-mask-path', type=str, default='./mask/undersampling_mask/mask_8.00x_acs24.mat', help='undersampling mask')
#欠采样问题
parser.add_argument('--s-mask-up-path', type=str, default='./mask/selecting_mask/mask_2.00x_acs16.mat', help='selection mask in up network')
parser.add_argument('--s-mask-down-path', type=str, default='./mask/selecting_mask/mask_2.50x_acs16.mat', help='selection mask in down network')
parser.add_argument('--train-sample-rate', '-trsr', type=float, default=0.06, help='sampling rate of training data')
parser.add_argument('--val-sample-rate', '-vsr', type=float, default=0.02, help='sampling rate of validation data')
parser.add_argument('--test-sample-rate', '-tesr', type=float, default=0.02, help='sampling rate of test data')
# save path
parser.add_argument('--model-save-path', type=str, default='./checkpoints/', help='save path of trained model')
parser.add_argument('--loss-curve-path', type=str, default='./runs/loss_curve/', help='save path of loss curve in tensorboard')
# others
parser.add_argument('--mode', '-m', type=str, default='train', help='whether training or test model, value should be set to train or test')
parser.add_argument('--pretrained', '-pt', type=bool, default=False, help='whether load checkpoint')

#新网络参数
parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

def create_logger():
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s:\t%(message)s')
    stream_formatter = logging.Formatter('%(levelname)s:\t%(message)s')

    file_handler = logging.FileHandler(filename='logger.txt', mode='a+', encoding='utf-8')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

#初始化是否去掉
def init_weights(net, init_type='xavier', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method {} is not implemented.'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class EarlyStopping:
    def __init__(self, patience=50, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        score = -metrics if loss else metrics
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def forward(mode, rank, model, dataloader, criterion, optimizer, log, args):
    count=0
    assert mode in ['train', 'val', 'test']
    loss, psnr, ssim = 0.0, 0.0, 0.0
    t = tqdm(dataloader, desc=mode + 'ing', total=int(len(dataloader))) if rank == 0 else dataloader
    for iter_num, data_batch in enumerate(t):
        k_und=data_batch[0].to(rank, non_blocking=True)
        und_mask=data_batch[1].to(rank, non_blocking=True)
        im_gt = data_batch[2].to(rank, non_blocking=True)
        
        ks=np.ones((256,256,1))
        select_mask_up,select_mask_down=uniform_selection(ks,und_mask)
        select_mask_up=np.expand_dims(select_mask_up,0).repeat(args.batch_size,axis=0)#此时mask的尺寸为（batch，256,256）
        select_mask_down=np.expand_dims(select_mask_down,0).repeat(args.batch_size,axis=0)#此时mask的尺寸为（batch，256,256）
        '''
        und_mask.shape: (1, 1, 256, 256)
        select_mask_up.shape: (1, 256, 256)
        select_mask_down.shape: (1, 256, 256)
        '''
        # und_mask=torch.from_numpy(und_mask)
        select_mask_up=torch.from_numpy(select_mask_up)
        select_mask_down=torch.from_numpy(select_mask_down)

        # und_mask=und_mask.permute(0,2,3,1)#1,256,256,1
        select_mask_up=select_mask_up.unsqueeze(3)#1,256,256,1
        select_mask_down=select_mask_down.unsqueeze(3)

        # und_mask=und_mask.repeat(1,1,1,2)
        select_mask_up=select_mask_up.repeat(1,1,1,2)
        select_mask_down=select_mask_down.repeat(1,1,1,2) #1,256,256,2

        select_mask_up=select_mask_up.to(rank)
        select_mask_down=select_mask_down.to(rank)

        # und_mask = und_mask.float()
        select_mask_up = select_mask_up.float()
        select_mask_down = select_mask_down.float()
        
        im_gt=im_gt.contiguous()
        select_mask_down=select_mask_down.permute(0,3,1,2)
        select_mask_up=select_mask_up.permute(0,3,1,2)
        down_img,down_kspace=np_undersample(pseudo2complex(k_und)[0],select_mask_down[0][0])
       
        up_img,up_kspace=np_undersample(pseudo2complex(k_und)[0],select_mask_up[0][0])
        # print('up_kspace.shape:',up_kspace.shape)#(1, 2, 256, 256)
        net_img_up=kspace2image(up_kspace)
        net_img_down=kspace2image(down_kspace)
        # net_img_down
        print('`````````````````````````k_und.shape:',k_und.shape)
        print('`````````````````````````k_und.dtype:',k_und.dtype)
        under_img = kspace2image(pseudo2complex(k_und))
        under_kspace = k_und

        if mode == 'test':
            net_img_up = net_img_down = under_img
            select_mask_up = select_mask_down = und_mask
            #使用网络部分
        
        #此部分可能需要改变
        net_img_up=complex2pseudo(net_img_up)
        net_img_down=complex2pseudo(net_img_down)
        under_img=complex2pseudo(under_img)
        # net_img_up=torch.from_numpy(net_img_up)
        # net_img_down=torch.from_numpy(net_img_down)
        
        net_img_up=net_img_up.to(rank)
        net_img_down=net_img_down.to(rank)
        net_img_up=net_img_up.to(torch.float32)
        net_img_down=net_img_down.to(torch.float32)
        
        net_img_up=net_img_up.squeeze(0)
        net_img_down=net_img_down.squeeze(0)
        net_img_up=net_img_up.repeat(args.batch_size,1,1,1)
        net_img_down=net_img_down.repeat(args.batch_size,1,1,1)
        print('under_img.shape:',under_img.shape)#under_img.shape: torch.Size([1, 2, 2, 256, 256])
        print('under_img.dtype:',under_img.dtype)
        print('net_img_up.shape:',net_img_up.shape)
        print('und_mask.shape:',und_mask.shape)
        output_up, output_down,output_mid = model(net_img_up.contiguous(), net_img_down.contiguous(),under_img.contiguous(),und_mask.contiguous())#output class 设成1
        # count=iter_num+count
   
        # save_image(net_img_up1,f'{results_save_path}/{count}.png')
      
        # output_up, output_down,output_mid = output_up.permute(0, 2, 3, 1).contiguous(), output_down.permute(0, 2, 3, 1).contiguous(),output_mid.permute(0, 2, 3, 1).contiguous()
        print('before_output_up.dtype:',output_up.dtype)
        print('before_output_up.shape:',output_up.shape)
        output_up_kspace = image2kspace(pseudo2complex(output_up))
        output_down_kspace = image2kspace(pseudo2complex(output_down))
        output_mid_kspace = image2kspace(pseudo2complex(output_mid))
        
        # output_up_kspace=output_up_kspace.permute(0,3,1,2)#1,2,256,256
        # output_down_kspace=output_down_kspace.permute(0,3,1,2)#1,2,256,256
        # output_mid_kspace=output_mid_kspace.permute(0,3,1,2)#1,2,256,256
        # output_up_kspace=output_up_kspace.squeeze(0)
        # output_down_kspace=output_down_kspace.squeeze(0)
        # output_mid_kspace=output_mid_kspace.squeeze(0)
        output_up_kspace=complex2pseudo(output_up_kspace)
        output_down_kspace=complex2pseudo(output_down_kspace)
        output_mid_kspace=complex2pseudo(output_mid_kspace)
        print('output_up_kspace.shape:',output_up_kspace.shape)
        und_mask=und_mask.unsqueeze(3)#1,256,256,1
        select_mask_up=select_mask_up.repeat(1,1,1,2)#（1,256,256,2）

        und_mask=und_mask.permute(0,3,1,2)
        und_mask=und_mask.repeat(1,2,1,1)

        #loss shape test
        print('loss_output_up_kspace_y1.shape:',output_up_kspace.shape)
        print('loss_output_down_kspace_y2.shape:',output_down_kspace.shape)
        print('loss_output_up_kspace_yu.shape:',output_mid_kspace.shape)
        print('loss_under_kspace_y.shape:',under_kspace.shape)
        
        

        y1,y2,yu,y=output_up_kspace,output_down_kspace,output_mid_kspace,under_kspace
        x1,x2,xu,x= output_up,output_down,output_mid,under_img
        # x1,x2,xu=x1.permute(0,2,3,1),x2.permute(0,2,3,1),xu.permute(0,2,3,1)
        # print('loss_output_up_x1.shape:',x1.shape)
        # print('loss_output_down_x2.shape:',x2.shape)
        # print('loss_output_mid_xu.shape:',xu.shape)
        # print('loss_under_img_x.shape:',x.shape)
        batch_loss=cal_loss(y,y1,y2,yu,x,x1,x2,xu)
        if mode == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        else:
            #评估指标
            ssim+=compute_ssim(output_up,im_gt)
            psnr+=compute_psnr(output_up,im_gt)
           
        loss += batch_loss.item()
    loss /= len(dataloader)
    log.append(loss)
    if mode == 'train':
        curr_lr = optimizer.param_groups[0]['lr']
        log.append(curr_lr)
    else:
        psnr /= len(dataloader)
        ssim /= len(dataloader)
        log.append(psnr)
        log.append(ssim)
    return log


def solvers(rank, ngpus_per_node, args):
    if rank == 0:
        logger = create_logger()
        logger.info('Running distributed data parallel on {} gpus.'.format(args.world_size))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size, rank=rank)
    # set initial value
    start_epoch = 0
    best_ssim = 0.0
    # model
    model = Network(num_layers=args.num_layers, rank=rank,bilinear=args.bilinear)#默认值输入
    # whether load checkpoint  模型参数保存 改成共享参数可能会变化
    if args.pretrained or args.mode == 'test':
        model_path = os.path.join(args.model_save_path, 'best_checkpoint.pth.tar')
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path, map_location='cuda:{}'.format(rank))
        start_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        args.lr = lr
        best_ssim = checkpoint['best_ssim']
        model.load_state_dict(checkpoint['model'])
        if rank == 0:
            logger.info('Load checkpoint at epoch {}.'.format(start_epoch))
            logger.info('Current learning rate is {}.'.format(lr))
            logger.info('Current best ssim in train phase is {}.'.format(best_ssim))
            logger.info('The model is loaded.')
    elif args.use_init_weights:
        init_weights(model, init_type=args.init_type, gain=args.gain)
        if rank == 0:
            logger.info('Initialize model with {}.'.format(args.init_type))
    model = model.to(rank)
    # model = DDP(model, device_ids=[rank])#为了实现单GPU进行训练

    # criterion, optimizer, learning rate scheduler
    #损失函数部分 根据论文进行改变
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if not args.pretrained:
        warm_up = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 1
        scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up)
    scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.3, patience=20)
    early_stopping = EarlyStopping(patience=50, delta=1e-5)

    #数据集部分
    dataset1 = FastmriKnee('/data/liuchun/dual_domain/data/knee_singlecoil_1000.npz')
    dataset=DatasetReconMRI(dataset1)
    train_loader,val_loader,test_loader=build_loader(dataset,args.batch_size)

    # test step  数据集部分更改
    if args.mode == 'test':

        if rank == 0:
            logger.info('The size of test dataset is {}.'.format(len(test_set)))
            logger.info('Now testing {}.'.format(args.exp_name))
        model.eval()
        with torch.no_grad():
            test_log = []
            start_time = time.time()
            print('test_loader:',len(test_loader))
            test_log = forward('test', rank, model, test_loader, criterion, optimizer, test_log, args)
            test_time = time.time() - start_time
        # test information
        test_loss = test_log[0]
        test_psnr = test_log[1]
        test_ssim = test_log[2]
        if rank == 0:
            logger.info('time:{:.5f}s\ttest_loss:{:.7f}\ttest_psnr:{:.5f}\ttest_ssim:{:.5f}'.format(test_time, test_loss, test_psnr, test_ssim))
        return

    if rank == 0:
        # logger.info('The size of training dataset and validation dataset is {} and {}, respectively.'.format(len(train_set), len(val_set)))
        logger.info('Now training {}.'.format(args.exp_name))
        writer = SummaryWriter(args.loss_curve_path)
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        # train_sampler.set_epoch(epoch)
        train_log = [epoch]
        epoch_start_time = time.time()
        model.train()
        print('train_loader:',len(train_loader))
        train_log = forward('train', rank, model, train_loader, criterion, optimizer, train_log, args)
        model.eval()
        with torch.no_grad():
            train_log = forward('val', rank, model, val_loader, criterion, optimizer, train_log, args)
        epoch_time = time.time() - epoch_start_time
        # train information
        epoch = train_log[0]
        train_loss = train_log[1]
        lr = train_log[2]
        val_loss = train_log[3]
        val_psnr = train_log[4]
        val_ssim = train_log[5]

        is_best = val_ssim > best_ssim
        best_ssim = max(val_ssim, best_ssim)
        if rank == 0:
            logger.info('epoch:{:<8d}time:{:.5f}s\tlr:{:.8f}\ttrain_loss:{:.7f}\tval_loss:{:.7f}\tval_psnr:{:.5f}\t'
                        'val_ssim:{:.5f}'.format(epoch, epoch_time, lr, train_loss, val_loss, val_psnr, val_ssim))
            writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            # save checkpoint
            checkpoint = {
                'epoch': epoch,
                'lr': lr,
                'best_ssim': best_ssim,
                'model': model.state_dict()
                # 'model': model.module.state_dict()
            }
            if not os.path.exists(args.model_save_path):
                os.makedirs(args.model_save_path)
            model_path = os.path.join(args.model_save_path, 'checkpoint.pth.tar')
            best_model_path = os.path.join(args.model_save_path, 'best_checkpoint.pth.tar')
            torch.save(checkpoint, model_path)
            if is_best:
                shutil.copy(model_path, best_model_path)
        # scheduler
        if epoch <= args.warmup_epochs and not args.pretrained:
            scheduler_wu.step()
        scheduler_re.step(val_ssim)
        early_stopping(val_ssim, loss=False)
        if early_stopping.early_stop:
            if rank == 0:
                logger.info('The experiment is early stop!')
                im_gtl=im_gt[0,0,:,:]
                save_image(im_gtl,f'{results_save_path}/{4}.png')
            break
    if rank == 0:
        writer.close()
    return


def main():
    args = parser.parse_args()
    args.world_size = args.nodes * args.gpus
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.multiprocessing.spawn(solvers, nprocs=args.gpus, args=(args.gpus, args))


if __name__ == '__main__':
    main()
