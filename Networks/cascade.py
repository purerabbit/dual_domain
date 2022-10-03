import torch
import torch.nn as nn
from data.utils import *


class DataConsistencyLayer(nn.Module):
    """
    This class support different types k-space data consistency
    """

    def __init__(self, is_data_fidelity=False):
        super().__init__()
        self.is_data_fidelity = is_data_fidelity
        if is_data_fidelity:
            self.data_fidelity = nn.Parameter(torch.randn(1))

    def data_consistency(self, k, k0, mask):
        """
        :param k: input k-space (reconstructed kspace, 2D-Fourier transform of im)
        :param k0: initially sampled k-space
        :param mask: sampling pattern
        """
        if self.is_data_fidelity:
            v = self.is_data_fidelity
            k_dc = (1 - mask) * k + mask * (k + v * k0 / (1 + v))
        else:
            k_dc = (1 - mask) * k + mask * k0
        return k_dc

    def forward(self, im, k0, mask):
        """
        im   - Image in pseudo-complex [B, C=2, H, W]
        k0   - original under-sampled Kspace in pseudo-complex [B, C=2, H, W]
        mask - mask for Kspace in Real [B, H, W]
        """
        # mask need to add one axis to broadcast to pseudo-complex channel
        # print('dc_im.shape:',im.shape)
        k = image2kspace(pseudo2complex(im))  # [B, H, W] Complex
        k0 = pseudo2complex(k0)
        k_dc = self.data_consistency(k, k0, mask)  # [B, H, W] Complex
        im_dc = complex2pseudo(kspace2image(k_dc))  # [B, C=2, H, W]

        return im_dc


class CascadeMRIReconstructionFramework(nn.Module):
    def __init__(self, recon_net: nn.Module, n_cascade: int):
        super().__init__()
        self.recon_net = recon_net
        self.n_cascade = n_cascade

        assert n_cascade > 0
        dc_layers = [DataConsistencyLayer() for _ in range(n_cascade)]
        self.dc_layers = nn.ModuleList(dc_layers)

    def forward(self, im_und, mask):
        # B, C, H, W = k_und.shape
        B, C, H, W = im_und.shape
        assert C == 2
        print('cascade-mask.shape:',mask.shape)#torch.Size([1, 2, 256, 256])
        assert (B, H, W) == tuple(mask.shape)

        # im_und = complex2pseudo(kspace2image(pseudo2complex(k_und)))
        # print('casca_im_und.shape:',im_und.shape)
        k_und = complex2pseudo(image2kspace(pseudo2complex(im_und)))
        # print('after_complexpesudo_im_und.shape:',im_und.shape)
        im_recon = im_und
        # print('init_im_recon.shape:',im_recon.shape)#init_im_recon.shape: torch.Size([1, 2, 256, 256])
        for dc_layer in self.dc_layers:
            # print('recon_net_im_recon.shape:',im_recon.shape)#torch.Size([1, 2, 2, 256, 256])
            im_recon = self.recon_net(im_recon)
            im_recon = dc_layer(im_recon, k_und, mask)
        return im_recon