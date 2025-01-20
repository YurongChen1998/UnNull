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





def UnNull_MSFA(meas, Phi, LRHSI, X_truth_tensor, Filter_matrix, truth_tensor):
    torch.backends.cudnn.benchmark = True
    _, B, _, _ = LRHSI.shape
    iter_num = 2000
    best_loss = float('inf')
    loss_l1 = torch.nn.L1Loss().to(device)
    loss_l2 = torch.nn.MSELoss().to(device)
    im_net = model_load(B)
   
    save_model_weight = False
    if os.path.exists('Results/model_init_weights.pth'):
        ckpt = torch.load('Results/model_init_weights.pth')
        del ckpt['encoder0.0.1.weight']
        del ckpt['skip0.0.1.weight']
        del ckpt['recon_head.1.weight']
        del ckpt['recon_head.1.bias']
        im_net[0].load_state_dict(ckpt, strict=False)
        print('----------------------- Load inital model weights -----------------------')
        save_model_weight = False
        
    im_net[0].train()
    net_params = list(im_net[0].parameters())
    optimizer = torch.optim.Adam([{'params': net_params, 'lr': 1e-3}])
    
    begin_time = time.time()
    for idx in range(iter_num):
        net_out = im_net[0](LRHSI, Phi, Filter_matrix)
        model_out = net_out + LRHSI
        model_out_x = model_out.repeat([Filter_matrix.shape[0], 1, 1, 1])
        model_out_x = (model_out_x.permute(0, 2, 3, 1) * Filter_matrix).sum(3)
        
        pred_meas = A(model_out_x.permute(1, 2, 0), Phi)
        loss = loss_l1(meas, pred_meas)
        loss_tv = calculate_stv(model_out.squeeze(0).permute(1, 2, 0))
        loss += 0.5*loss_tv
        
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