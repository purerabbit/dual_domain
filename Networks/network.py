
from .unet.unet_model import UNet#用此方式可以实现包的导入

#从不同文件夹下导入包
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cascade import CascadeMRIReconstructionFramework
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
        recon_net=UNet(n_channels=2, n_classes=2, bilinear=self.bilinear),
        n_cascade=5
)

    def forward(self, under_img_up, under_img_down,under_img,mask):
        output_up = self.net(under_img_up,mask)
        output_mid=self.net(under_img,mask)
        output_down = self.net(under_img_down,mask)
        return output_up, output_down,output_mid
