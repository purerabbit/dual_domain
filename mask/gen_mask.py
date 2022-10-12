import numpy as np
import numpy as np
import sys
from numpy.lib.stride_tricks import as_strided
import torch


def norm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor : It can be in image space or k-space.
    axes :  The default is (0, 1, 2).
    keepdims : The default is True.

    Returns
    -------
    tensor : applies l2-norm . #默认求2范数

    """
    for axis in axes:#分别求出需要的范数 保存到tensor中
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)#求范数

    if not keepdims: return tensor.squeeze()#将输入张量形状中的1去除并返回

    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace : nrow x ncol x ncoil.
    axes :  The default is (1, 2, 3).

    Returns
    -------
    the center of the k-space

    """

    center_locs = norm(kspace, axes=axes).squeeze()#何种数据类型，返回的是tensor 对谁计算之后的tensor？

    return np.argsort(center_locs)[-1:]#np.argsort 生成把center_locs从小到大排序的下标
def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/tensor for mapping ind.

    Returns
    -------
    list of >=2D indices containing non-zero locations

    """

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)



#生成模拟欠采的mask

def cartesian_mask(shape, acc, sample_n, centred=True):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)  # ? this line may be wrong. shape is (Nslice, Nx, Ny, Ntime)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N=1  #表示份数
    Nx, Ny = shape[0], shape[1]
    # N, Nx, Ny = int(np.prod([shape[0], shape[-1]])), shape[1], shape[2]
    
    pdf_x = normal_pdf(Nx, 0.5 / (Nx / 10.) ** 2)
    lmda = Nx / (2. * acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1. / Nx

    if sample_n:
        pdf_x[Nx // 2 - sample_n // 2:Nx // 2 + sample_n // 2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Nx // 2 - sample_n // 2:Nx // 2 + sample_n // 2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    '''
    print('mask.shape:',mask.shape)#(1,256,256)
    print('shape[0]:',shape[0])#256
    print('shape[-1]:',shape[-1])#256
    print('Nx:',Nx)#256
    print('Ny:',Ny)#256
    '''
    
    # mask = mask.reshape((shape[0], shape[-1], Nx, Ny))  没有时间维度 不用 reshape
    # mask = np.transpose(mask, [0, 2, 3, 1])#作用是改变序列  与permute的作用一样

    #如果后续出现了问题 只需保证输出的内容与kspace可以直接相乘即可
    #后续kspace如何傅里叶变换除了问题可以引入fastmri包中的相关函数
    #这句话是否make sense?
    if not centred:
        mask = np.fft.ifftshift(mask, axes=(1, 2))
    return mask

#生成划分mask
'''input_data: input k-space, nrow x ncol x ncoil
    input_mask: input mask, nrow x ncol '''

def uniform_selection( input_data, input_mask, num_iter=1):#input_data->kspace   input_mask->mask_init
    # print('input_mask.shape:',input_mask.shape)
    # print('need to be (256,256,1)')
    input_mask=input_mask.permute(1,2,0)
    input_mask=np.squeeze(input_mask)
    # print('input_data.shape:',input_data.shape) #(256,256,1)
    # print('input_mask.shape:',input_mask.shape) #(256,256)
    small_acs_block=(4,4)
    rho=0.5     
    nrow, ncol = input_data.shape[0], input_data.shape[1]#行列 根据input的数据来定

    center_kx = int(find_center_ind(input_data, axes=(1, 2)))
    center_ky = int(find_center_ind(input_data, axes=(0, 2)))

    if num_iter == 0:
        print(f'\n Uniformly random selection is processing, rho = {rho:.2f}, center of kspace: center-kx: {center_kx}, center-ky: {center_ky}')
    #input_mask如何确定？
    temp_mask = np.copy(input_mask.cpu())
    temp_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
    center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 0

    pr = np.ndarray.flatten(temp_mask)
    ind = np.random.choice(np.arange(nrow * ncol),
                            size=np.int(np.count_nonzero(pr) * rho), replace=False, p=pr / np.sum(pr))
    
    #实现原来欠采样的mask上进行采样
    [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))
    # print('input_mask:')
    # assert isinstance(input_mask,torch.Tensor)
    inpumak=input_mask.cpu().data.numpy()
    loss_mask = np.zeros_like(inpumak)
    # print('input_mask.shape:',input_mask.shape)#input_mask.shape: (1, 256, 256)
    
    loss_mask[ind_x, ind_y] = 1

    trn_mask = input_mask.cpu().data.numpy() - loss_mask

    return trn_mask, loss_mask


