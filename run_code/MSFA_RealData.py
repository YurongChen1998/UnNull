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
from optim_code.model_MSFA import UnNull_Real_MSFA
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

       

    
def main_MSFA_Realdata(data_name):
    #----------------------- Data Configuration -----------------------#
    print('\n')
    print('Test Scene:', data_name)
    
    dataset_dir = '../../Dataset/Real_Data/'
    result_dir = './Results/' + data_name + '/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    matfile = dataset_dir + data_name + '.mat'
    h, w, nC = 1000, 1000, 25
    data = sio.loadmat(matfile)

    Phi = torch.from_numpy(data['SMP_seq']).float().to(device)
    meas_ = torch.from_numpy(data['I_MOS_seq']).float().to(device) / 255
    meas = meas_
    LRHSI = meas_.unsqueeze(2).repeat(1, 1, nC)
    LRHSI_new = torch.zeros_like(LRHSI)
    for i in range(nC):
        LRHSI_new[:, :, i] = tensor_weight_conv(LRHSI[:, :, i].cpu() * Phi[:, :, i].cpu(), nC).to(device)
    LRHSI = LRHSI_new.unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
    truth_tensor = LRHSI

    print('                                                        ')
    print('--------- Data Size:', truth_tensor.shape)
    print('--------- Phi Size:', Phi.shape)
    print('--------- RAW Size:', meas.shape)
    print('                                                        ')
    
    #------------------------- Training Model -------------------------#
    recon = UnNull_Real_MSFA(meas, Phi, LRHSI, truth_tensor)
    sio.savemat('{}/{}.mat'.format(result_dir, data_name), {'LRHSI':LRHSI.squeeze().cpu().numpy(), 'img':recon.cpu().numpy()})


