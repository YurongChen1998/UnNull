##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                           Hunan University                           ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
import numpy as np
from func import *
from numpy import *
import scipy.io as sio
from torchmetrics import SpectralAngleMapper, ErrorRelativeGlobalDimensionlessSynthesis
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(data_name):
    #----------------------- Data Configuration -----------------------#
    print('\n')
    print('Test Scene:', data_name)
    
    dataset_dir = 'E:/Yurong_3080/2024-03-30-Backup_3080/Working_Yurong/Working_Spectral_Demosaicing//Dataset/5x5/Filter_Sampling/CAVE_balloons.mat'
    data_truth = torch.from_numpy(sio.loadmat(dataset_dir)['Img'] / 255)
    
    recon_dir = 'E:/Yurong_3080/2024-03-30-Backup_3080/Working_Yurong/Working_Spectral_Demosaicing/UnNull/Final_Model_2024-01-06/Results/5x5/Filter_Sampling/CAVE_balloons/LSD.mat'
    recon = torch.from_numpy(sio.loadmat(recon_dir)['img'])

    sam = SpectralAngleMapper()
    vrecon = recon.double().cpu()
    ssim_ = calculate_ssim(data_truth, vrecon)
    psnr_ = calculate_psnr(data_truth, vrecon)
    sam_ = sam(torch.unsqueeze(data_truth.permute(2, 0, 1), 0).double(), torch.unsqueeze(vrecon.permute(2, 0, 1), 0).double())
    print('PSNR {:2.3f}, ---------, SSIM {:2.3f}, ---------, SAM {:2.3f}'.format(psnr_, ssim_, sam_))


    '''
    x = vrecon.clamp_(0, 1).numpy()
    data_truth = data_truth.numpy()
    psnr_ = [cpsnr(x[..., kv], data_truth[..., kv]) for kv in range(28)]
    ssim_ = [cssim(x[..., kv], data_truth[..., kv]) for kv in range(28)]
    print('---------- PNSR:', np.mean(psnr_), '---------- SSIM:', np.mean(ssim_))
    '''


data_list = ['scene05']
for file_name in data_list:
    main(file_name)
