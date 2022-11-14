import torch
import torch.nn as nn
from data.utils import *
from .unet.unet_model import UNet#用此方式可以实现包的导入

class DataConsistencyLayer(nn.Module):
    """
    This class support different types k-space data consistency
    """

    def __init__(self, is_data_fidelity=True):
        super().__init__()
        self.is_data_fidelity = is_data_fidelity
        self.cnn=UNet(n_channels=2, n_classes=2, bilinear=False)
        # self.cnn_real=UNet(n_channels=1, n_classes=1, bilinear=False)
        # self.cnn_img=UNet(n_channels=1, n_classes=1, bilinear=False)
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
        k=complex2pseudo(image2kspace(pseudo2complex(im_recon)))
        if self.is_data_fidelity:
            k_dc =  mask*k - k0
            k_dc=mask*k_dc
            im_dc = complex2pseudo(kspace2image(pseudo2complex(k_dc)))
            # im_dc_real,im_dc_img=im_dc.split(1,dim=1)
            # im_dc_real = im_dc_real*self.data_fidelity
            # im_dc_img = im_dc_img*self.data_fidelity
            # im_dc=torch.cat((im_dc_real,im_dc_img),dim=1)
            im_dc = im_dc*self.data_fidelity  #虚部和实部使用不同的可学习参数？
            # print('self.data_fidelity:',self.data_fidelity)
            # im_recon_real,im_recon_img=im_recon.split(1,dim=1)
            # im_recon_real=self.cnn_real(im_recon_real)
            # im_recon_img=self.cnn_img(im_recon_img)
            im_cnn= self.cnn(im_recon)
<<<<<<< HEAD
            # im_cnn=torch.cat((im_recon_real,im_recon_img),dim=1)
            im_recon=im_recon-im_dc-im_cnn #按照公式将-改成+
=======
            im_recon=im_recon-im_dc+im_cnn #按照公式将-改成+
>>>>>>> 9451285a496f76460e62711338dd5a1412bdca4c
           
        else:
            print('wrong!,set is_data_fidelity=True')
            
        return im_recon


class CascadeMRIReconstructionFramework(nn.Module):
    def __init__(self,  n_cascade: int):
        super().__init__()
      
        self.n_cascade = n_cascade

        assert n_cascade > 0
        VnMriReconCell = [DataConsistencyLayer() for _ in range(n_cascade)]
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