##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                          Hunan University                            ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import torch
import time
import scipy.io as sio
from func import *
from models.model_loader import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')





def UnNull_MSFA(meas, Phi, LRHSI, truth_tensor):
    torch.backends.cudnn.benchmark = True
    _, _, B = Phi.shape
    iter_num = 500
    best_loss = float('inf')
    loss_l1 = torch.nn.L1Loss().to(device)
    loss_l2 = torch.nn.MSELoss().to(device)
    im_net = low_rank_model_load(B)
   
    save_model_weight = False
    if os.path.exists('Results/model_init_weights.pth'):
        im_net[0].load_state_dict(torch.load('Results/model_init_weights.pth'))
        print('----------------------- Load inital model weights -----------------------')
        save_model_weight = False
        
    im_net[0].train()
    net_params = list(im_net[0].parameters())
    optimizer = torch.optim.Adam([{'params': net_params, 'lr': 1e-3}])
    
    begin_time = time.time()
    for idx in range(iter_num):
        net_out = im_net[0](LRHSI, Phi)
        model_out = net_out + LRHSI
        pred_meas = A(model_out.squeeze(0).permute(1, 2, 0), Phi)
        loss = loss_l1(meas, pred_meas)
        loss_tv = calculate_stv(model_out.squeeze(0).permute(1, 2, 0))
        loss += 6*loss_tv
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
             
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_hs_recon = model_out.detach()
            if save_model_weight == True:
                torch.save(im_net[0].state_dict(), 'Results/model_weights.pth')
        
        if (idx+1)%50==0:
            PSNR = calculate_psnr(truth_tensor, model_out.squeeze(0))
            print('Iter {}, x_loss:{:.3f}, tv_loss:{:.3f}, PSNR:{:.2f}'.format(idx+1, loss.item(), 1000*loss_tv.item(), PSNR))       
                    
    end_time = time.time()
    print('-------------- Finished----------, running time {:.1f} seconds.'.format(end_time - begin_time))
    return best_hs_recon.squeeze(0).permute(1, 2, 0)







def UnNull_Real_MSFA(meas, Phi, LRHSI, truth_tensor):
    torch.backends.cudnn.benchmark = True
    _, _, B = Phi.shape
    iter_num = 40
    best_loss = float('inf')
    loss_l1 = torch.nn.L1Loss().to(device)
    loss_l2 = torch.nn.MSELoss().to(device)
    im_net = low_rank_model_load(B)
   
    save_model_weight = False
    if os.path.exists('Results/model_init_weights.pth'):
        im_net[0].load_state_dict(torch.load('Results/model_init_weights.pth'))
        print('----------------------- Load inital model weights -----------------------')
        save_model_weight = False
        
    im_net[0].train()
    net_params = list(im_net[0].parameters())
    optimizer = torch.optim.Adam([{'params': net_params, 'lr': 1e-3}])
    
    begin_time = time.time()
    for idx in range(iter_num):
        model_out = im_net[0](LRHSI, Phi)
        pred_meas = A(model_out.squeeze(0).permute(1, 2, 0), Phi)
        loss = loss_l1(meas, pred_meas)
        loss_tv = calculate_stv(model_out.squeeze(0).permute(1, 2, 0))
        loss += 100*loss_tv
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
             
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_hs_recon = model_out.detach()
            if save_model_weight == True:
                torch.save(im_net[0].state_dict(), 'Results/model_weights.pth')
        
        if (idx+1)%5==0:
            PSNR = calculate_psnr(truth_tensor, model_out.squeeze(0))
            print('Iter {}, x_loss:{:.3f}, tv_loss:{:.3f}, PSNR:{:.2f}'.format(idx+1, loss.item(), 1000*loss_tv.item(), PSNR))
                    
    end_time = time.time()
    print('-------------- Finished----------, running time {:.1f} seconds.'.format(end_time - begin_time))
    return best_hs_recon.squeeze(0).permute(1, 2, 0)
