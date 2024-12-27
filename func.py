##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                          Hunan University                            ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage import img_as_ubyte
import sys
import numpy as np
from pytorch_msssim import ssim
from scipy.signal import convolve2d
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def A(data, Phi):
    return torch.sum(data * Phi, 2)


def At(meas, Phi):
    meas = torch.unsqueeze(meas, 2).repeat(1, 1, Phi.shape[2])
    return meas * Phi


def shift(inputs, step):
    [h, w, nC] = inputs.shape
    output = torch.zeros((h, w+(nC - 1)*step, nC)).to(device)
    for i in range(nC):
        output[:, i*step : i*step + w, i] = inputs[:, :, i]
    del inputs
    return output


def shift_back(inputs, step):
    [h, w, nC] = inputs.shape
    for i in range(nC):
        inputs[:, :, i] = torch.roll(inputs[:, :, i], (-1)*step*i, dims=1)
    output = inputs[:, 0 : w - step*(nC - 1), :]
    return output

def calculate_tv(x):
    N = x.shape
    idx = torch.arange(1, N[0]+1)
    idx[-1] = N[0]-1
    ir = torch.arange(1, N[1]+1)
    ir[-1] = N[1]-1

    x1 = x[:,ir,:] - x
    x2 = x[idx,:,:] - x
    tv = (x1)**2 + (x2)**2
    return torch.mean(torch.mean(tv, 2))

def calculate_stv(x):
    N = x.shape
    ir = torch.arange(1, N[2]+1)
    ir[-1] = N[2]-1

    x1 = x[:,:,ir] - x
    tv = torch.mean((x1)**2)
    return tv
    
def ssim_(data, recon):
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    data = data.astype(np.float64)
    recon = recon.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(data, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(recon, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(data ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(recon ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(data * recon, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(data, recon, border=0):
    if not data.shape == recon.shape:
        raise ValueError('Data size must have the same dimensions!')
    if not data.dtype == recon.dtype:
        data, recon = data.float(), recon.float()
        
    h, w = data.shape[:2]
    data = data[border:h - border, border:w - border]
    recon = recon[border:h - border, border:w - border]
    if data.ndim == 2:
        return ssim_(data, recon)
    elif data.ndim == 3:
        return ssim(torch.unsqueeze(data, 0).permute(3, 0, 1, 2), torch.unsqueeze(recon, 0).permute(3, 0, 1, 2), data_range=1).data
     

def calculate_psnr(data, recon):
    mse = torch.mean((recon - data)**2)
    if mse == 0:
        return 100
    Pixel_max = 1.0 
    return 20 * torch.log10(Pixel_max / torch.sqrt(mse))


def clip(x):
    if not isinstance(x, np.ndarray):
        x = x.clone().cpu().numpy()
    return x.clip(0., 1.)


def ssim_index(im1, im2):
    if im1.ndim == 2:
        out = structural_similarity(im1, im2, data_range=255, gaussian_weights=True,
                                                    use_sample_covariance=False, multichannel=False)                                        
    elif im1.ndim == 3:
        out = structural_similarity(im1, im2, data_range=255, gaussian_weights=True,
                                                     use_sample_covariance=False, multichannel=True)
    else:
        sys.exit('Please input the corrected images')
    return out


def cssim(img, img_clean):
    if isinstance(img, torch.Tensor):
        img = img.data.cpu().numpy()
    if isinstance(img_clean, torch.Tensor):
        img_clean = img_clean.data.cpu().numpy()
    img = img_as_ubyte(img)
    img_clean = img_as_ubyte(img_clean)
    SSIM = ssim_index(img, img_clean)
    return SSIM


def cpsnr(img, img_clean):
    if isinstance(img, torch.Tensor):
        img = img.data.cpu().numpy()
    if isinstance(img_clean, torch.Tensor):
        img_clean = img_clean.data.cpu().numpy()
    img = img_as_ubyte(img)
    img_clean = img_as_ubyte(img_clean)
    PSNR = peak_signal_noise_ratio(img, img_clean, data_range=255)
    return PSNR
    

def get_input(tensize, const=10.0):
    inp = torch.rand(tensize)/const
    inp = torch.autograd.Variable(inp, requires_grad=True).to(device)
    inp = torch.nn.Parameter(inp)
    return inp
    

def TV_denoiser(x, _lambda, n_iter_max):
    dt = 0.25
    N = x.shape
    idx = torch.arange(1, N[0]+1)
    idx[-1] = N[0]-1
    iux = torch.arange(-1, N[0]-1)
    iux[0] = 0
    ir = torch.arange(1, N[1]+1)
    ir[-1] = N[1]-1
    il = torch.arange(-1, N[1]-1)
    il[0] = 0
    p1 = torch.zeros_like(x)
    p2 = torch.zeros_like(x)
    divp = torch.zeros_like(x)

    for i in range(n_iter_max):
        z = divp - x*_lambda
        z1 = z[:,ir,:] - z
        z2 = z[idx,:,:] - z
        denom_2d = 1 + dt*torch.sqrt(torch.sum(z1**2 + z2**2, 2))
        denom_3d = torch.unsqueeze(denom_2d, 2).repeat(1, 1, N[2])
        p1 = (p1+dt*z1)/denom_3d
        p2 = (p2+dt*z2)/denom_3d
        divp = p1-p1[:,il,:] + p2 - p2[iux,:,:]
    u = x - divp/_lambda;
    return u
    
    
def gap_denoise(meas, Phi, data_truth):
    #-------------- Initialization --------------#
    x0 = At(meas, Phi)
    meas_1 = torch.zeros_like(meas)
    iter_max = 300
    k = 0
    Phi_sum = torch.sum(Phi, 2)
    Phi_sum[Phi_sum==0] = 1
    x = x0

    # ---------------- Iteration ----------------#
    for idx in range(iter_max):
        x = x.to(device)   
        meas_b = A(x, Phi)
        meas_1 = meas_1 + (meas - meas_b)
        x = x + At((meas_1 - meas_b)/Phi_sum, Phi)
        x = shift_back(x, step=2)
        x = TV_denoiser(x, 30, 7)

        # --------------- Evaluation ---------------#
        if data_truth is not None:
            ssim_t = calculate_ssim(data_truth, x)
            psnr_t = calculate_psnr(data_truth, x)
            print('GAP-TV, iteration {}, '.format(k), 'PSNR {:2.2f} dB.'.format(psnr_t), 'SSIM:{:2.2f}'.format(ssim_t))

        x = shift(x, step=2)
        k = k + 1
    return shift_back(x, step=2)


def weight_conv(img, nC):
    if nC == 16:
        filters = [[1,2,3,4,3,2,1],
        [2,4,6,8,6,4,2],
        [3,6,9,12,9,6,3],
        [4,8,12,16,12,8,4],
        [3,6,9,12,9,6,3],
        [2,4,6,8,6,4,2],
        [1,2,3,4,3,2,1]]
        filters = np.array(filters) / 16.
    elif nC == 25:
        filters = [[1,2,3,4,5,4,3,2,1],
        [2,4,6,8,10,8,6,4,2],
        [3,6,9,12,15,12,9,6,3],
        [4,8,12,16,20,16,12,8,4],
        [5,10,15,20,25,20,15,10,5],
        [4,8,12,16,20,16,12,8,4],
        [3,6,9,12,15,12,9,6,3],
        [2,4,6,8,10,8,6,4,2],
        [1,2,3,4,5,4,3,2,1]]
        filters = np.array(filters) / 25.    
    return torch.from_numpy(convolve2d(img.numpy(), filters, mode='same'))
    
    
def tensor_weight_conv_v0(img, nC):
    if nC == 16:
        filters = [[1,2,3,4,3,2,1],
        [2,4,6,8,6,4,2],
        [3,6,9,12,9,6,3],
        [4,8,12,16,12,8,4],
        [3,6,9,12,9,6,3],
        [2,4,6,8,6,4,2],
        [1,2,3,4,3,2,1]]
        filters = torch.tensor(filters).to(device) / 16.
    elif nC == 25:
        filters = [[1,2,3,4,5,4,3,2,1],
        [2,4,6,8,10,8,6,4,2],
        [3,6,9,12,15,12,9,6,3],
        [4,8,12,16,20,16,12,8,4],
        [5,10,15,20,25,20,15,10,5],
        [4,8,12,16,20,16,12,8,4],
        [3,6,9,12,15,12,9,6,3],
        [2,4,6,8,10,8,6,4,2],
        [1,2,3,4,5,4,3,2,1]]
        filters = torch.tensor(filters).to(device) / 25.   
    img = img.unsqueeze(0).unsqueeze(0)
    filters = filters.unsqueeze(0).unsqueeze(0)
    img = F.conv2d(img, filters, padding=4)
    return img.squeeze(0).squeeze(0)
    

def tensor_weight_conv(img, nC):
    if nC == 16:
        kerel_size = 4
    elif nC == 25:
        kerel_size = 5
    pool_layer = nn.MaxPool2d(kerel_size, stride=kerel_size).to(device)

    img = img.unsqueeze(0).unsqueeze(0)
    img = pool_layer(img)
    img = F.interpolate(img, scale_factor=kerel_size, mode='bilinear')
    return img.squeeze(0).squeeze(0)

      
def weight_conv_denoise(img):
    filters = [[0.057, 0.125, 0.057],
    [0.125, 0.2725, 0.125],
    [0.057, 0.125, 0.057]]
    filters = np.array(filters)
    return torch.from_numpy(convolve2d(img.numpy(), filters, mode='same'))
