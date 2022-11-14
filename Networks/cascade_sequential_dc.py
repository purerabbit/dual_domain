import torch
import torch.nn as nn
from data.utils import *
from .unet.unet_v3 import UNet, UNetEncoder2d, UNetDecoder2d

class SingleCascade(nn.Module):
    """
    This class support different types k-space data consistency
    """

    def __init__(self, is_data_fidelity=True):
        super().__init__()
        self.is_data_fidelity = is_data_fidelity
        self.cnn=UNet(
            encoder=UNetEncoder2d(in_channels=2, base_channels=64, num_downs=2, downsample_size=(2, 2), is_batch_norm=False, n_conv_per_blk=2),
            decoder=UNetDecoder2d(out_channels = 2, base_channels=64, num_ups=2, upsample_size=(2, 2), is_batch_norm=False, n_conv_per_blk=2)
        )
        if is_data_fidelity:
            self.data_fidelity = nn.Parameter(torch.tensor(1.0,dtype=torch.float32))
            # self.data_fidelity = nn.Parameter(torch.ones((1,1),dtype=torch.float32))

    def forward(self, im_recon, k0, mask):
        """
        set is_data_fidelity=True to complete the formulation
        
        :param k: input k-space (reconstructed kspace, 2D-Fourier transform of im) complex
        :param k0: initially sampled k-space complex
        :param mask: sampling pattern
        """
        im_cnn= self.cnn(im_recon) + im_recon
        k_cnn = complex2pseudo(image2kspace(pseudo2complex(im_cnn)))
        v = self.is_data_fidelity
        k_dc = (1 - mask) * k_cnn + mask * (k_cnn + v * k0 / (1 + v))
        im_dc = complex2pseudo(kspace2image(pseudo2complex(k_dc)))
        im_recon = im_dc
            
        return im_recon


class CascadeMRIReconstructionFramework(nn.Module):
    def __init__(self,  n_cascade: int):
        super().__init__()
      
        self.n_cascade = n_cascade

        assert n_cascade > 0
        VnMriReconCell = [SingleCascade() for _ in range(n_cascade)]
        self.VnMriReconCells = nn.ModuleList(VnMriReconCell)

    def forward(self, im_und,init_unimg, mask):
        B, C, H, W = im_und.shape
        assert C == 2
        # assert (B, H, W) == tuple(mask.shape)
        k_und = complex2pseudo(image2kspace(pseudo2complex(init_unimg)))
        im_recon = im_und
        
        for vncell in self.VnMriReconCells:
            im_recon=vncell(im_recon, k_und, mask)
        return im_recon