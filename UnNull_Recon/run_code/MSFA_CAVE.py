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
import matplotlib.pyplot as plt
import torch.nn.functional as F
from optim_code.model_MSFA import UnNull_MSFA
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

       

        
def main_MSFA_CAVE(data_name):
    #----------------------- Data Configuration -----------------------#
    print('\n')
    print('Test Scene:', data_name)
    
    dataset_dir = '../../Dataset/5x5/Spectral_Recon/'
    result_dir = './Results/5x5/Spectral_Recon/' + data_name + '/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    matfile = dataset_dir + data_name + '.mat'
    h, w, nC, nB = 500, 500, 25, 31
    Filter_matrix = torch.from_numpy(sio.loadmat(dataset_dir + 'A_matrix.mat')['F']).float().to(device)
    data = sio.loadmat(matfile)
    Phi = torch.from_numpy(data['A']).float().to(device)
    
    data_truth = torch.from_numpy(data['Img'] / 255).float().to(device)
    truth_tensor = data_truth.permute(2, 0, 1).unsqueeze(0)
    X_data_truth = truth_tensor.repeat([nC, 1, 1, 1]).permute(0, 2, 3, 1) * Filter_matrix
    X_data_truth = X_data_truth.sum(3).permute(1, 2, 0)
    X_truth_tensor = X_data_truth.permute(2, 0, 1).unsqueeze(0)

    meas_3d = X_data_truth * Phi
    meas = meas_3d.sum(2)
    
    LRHSI_new = torch.zeros_like(Phi)
    for i in range(nC):
        LRHSI_new[:, :, i] = tensor_weight_conv(meas_3d[:, :, i] * Phi[:, :, i], nC).to(device)
    LRHSI_new = tensor_spectral_interpolate(LRHSI_new, nB)
       
    LRHSI = LRHSI_new.unsqueeze(0).permute(0, 3, 1, 2)
    PSNR = calculate_psnr(truth_tensor, LRHSI)

    print('                                                        ')
    print('--------- Start :')
    print('--------- LRHSI PSNR:', PSNR)
    print('                                                        ')
    
    '''
    Sparse_noise = np.random.choice((0, 1, 2), size=(meas.shape[0], meas.shape[1]), p=[0.99, 0.01/2., 0.01/2.])
    Gauss_noise = np.random.normal(loc=0.5, scale=0.5, size=(meas.shape[0], meas.shape[1]))
    meas = meas + 0.1*Gauss_noise
    meas[Sparse_noise == 1] = torch.max(meas)
    meas[Sparse_noise == 2] = 0
    meas = meas.float()
    '''


    #------------------------- Training Model -------------------------#
    recon = UnNull_MSFA(meas, Phi, LRHSI, X_truth_tensor, Filter_matrix, truth_tensor)
    sio.savemat('{}/UnNull.mat'.format(result_dir), {'img':recon.cpu().numpy()})
