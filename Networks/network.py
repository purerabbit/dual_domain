
from .unet.unet_model import UNet#用此方式可以实现包的导入

#从不同文件夹下导入包
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cascade_sequential_dc import CascadeMRIReconstructionFramework
from data.utils import *
#从不同文件夹下导入包


class ParallelNetwork(nn.Module):
   
    def __init__(self, num_layers, rank,bilinear=False):
        super(ParallelNetwork, self).__init__()
        self.num_layers = num_layers
        self.rank = rank
        self.bilinear=bilinear
        #cascade：input->recon_net: nn.Module, n_cascade: int
        #       forward_want:k_und, mask   mask is
        self.net = CascadeMRIReconstructionFramework(
<<<<<<< HEAD
            n_cascade=3  #the formor is 5
=======
            n_cascade=10  #the formor is 5
>>>>>>> 9451285a496f76460e62711338dd5a1412bdca4c
        )

    def forward(self, under_img_up, under_img_down,under_img,mask):
        # output_up = self.net(under_img_up,under_img,mask)
        output_mid=self.net(under_img,under_img,mask)
<<<<<<< HEAD
        # print('compute_psnr(output_mid,under_img):',compute_psnr(output_mid,under_img))
        # print('compute_ssim(output_mid,under_img):',compute_ssim(output_mid,under_img))
=======
>>>>>>> 9451285a496f76460e62711338dd5a1412bdca4c
        # output_down = self.net(under_img_down,under_img,mask)
        # return output_up, output_down,output_mid
        return  output_mid
