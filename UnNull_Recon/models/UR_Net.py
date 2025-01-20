from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import *
from func import *
             
            
class UR_Net(nn.Module):
    def __init__(self, input_depth, skip_n33d, filter_size_down, downsample_mode, upsample_mode, skip_n11, skip_n33u, filter_size_up, num_scales, num_output_channels):
        super(UR_Net, self).__init__()
        num_channels_down    = [skip_n33d]*num_scales
        num_channels_skip    = [skip_n11]*num_scales
        num_channels_up      = [skip_n33u]*num_scales
        
        filter_size_down     = [filter_size_down]*num_scales
        filter_size_up       = [filter_size_up]*num_scales
        filter_skip_size     = 1
        
        need_bias            = True
        pad                  = 'reflection'
        downsample_mode      = [downsample_mode]*num_scales
        act_fun              = 'LeakyReLU'
            
        i = 0
        self.encoder0 = nn.Sequential(
            conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]),
            act(act_fun),
            ca_conv(num_channels_down[i], num_channels_down[i]),
            act(act_fun),
            conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad),
            act(act_fun))
        self.up0 = nn.PixelShuffle(2)
        self.skip0 = nn.Sequential(
            conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad),
            act(act_fun),
            ca_conv(num_channels_skip[i], num_channels_skip[i]),
            act(act_fun))
        self.decoder0 = nn.Sequential(
            conv(num_channels_skip[i] + 8, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad),
            act(act_fun),
            ca_conv(num_channels_up[i], num_channels_up[i]),
            act(act_fun),
            conv(num_channels_up[i], num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad),
            act(act_fun))

        i = 1
        self.encoder1 = nn.Sequential(
            conv(num_channels_down[i-1], num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]),
            act(act_fun),
            ca_conv(num_channels_down[i], num_channels_down[i]),
            act(act_fun),
            conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad),
            act(act_fun))
        self.up1 = nn.PixelShuffle(2)
        self.skip1 = nn.Sequential(
            conv(num_channels_down[i-1], num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad),
            act(act_fun),
            ca_conv(num_channels_skip[i], num_channels_skip[i]),
            act(act_fun))
        self.decoder1 = nn.Sequential(
            conv(num_channels_skip[i] + 8, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad),
            act(act_fun),
            ca_conv(num_channels_up[i], num_channels_up[i]),
            act(act_fun),
            conv(num_channels_up[i], num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad),
            act(act_fun))
        self.recon_head = conv(num_channels_up[i], num_output_channels, 1, bias=need_bias, pad=pad)
        
    def forward_once(self, x):
        e0 = self.encoder0(x)
        s0 = self.skip0(x)
        e1 = self.encoder1(e0)
        s1 = self.skip1(e0)
        d1 = self.decoder1(torch.cat([self.up1(e1), s1], dim=1))
        d0 = self.decoder0(torch.cat([self.up0(d1), s0], dim=1))
        recon = self.recon_head(d0)
        return recon
          
    def forward(self, x, Phi, Filter_matrix):
        nC, h, w, nB = Filter_matrix.shape
        recon = self.forward_once(x)
        
        recon_x = recon.repeat([Filter_matrix.shape[0], 1, 1, 1])
        recon_x = (recon_x.permute(0, 2, 3, 1) * Filter_matrix).sum(3)
        pred_meas = recon_x.permute(1, 2, 0) * Phi
        LRHSI_new = torch.zeros([h, w, nC]).to(device)
        for i in range(nC):
            LRHSI_new[:, :, i] = tensor_weight_conv(pred_meas[:, :, i]*Phi[:, :, i], nC).to(device)
        LRHSI_new = tensor_spectral_interpolate(LRHSI_new, nB)
        LRHSI = LRHSI_new.unsqueeze(0).permute(0, 3, 1, 2)
        
        return recon - LRHSI
        
