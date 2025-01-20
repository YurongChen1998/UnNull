import torch
import torch.nn as nn
import os
from collections import OrderedDict
from models.UR_Net import UR_Net
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def model_load(rank):
    im_net = UR_Net(input_depth=rank, skip_n33d=32, filter_size_down=3, downsample_mode='stride',
                     upsample_mode = ['nearest', 'nearest'], skip_n11 = 16, skip_n33u = 32,
                     filter_size_up = 3, num_scales = 2, num_output_channels = rank).to(device)
    return [im_net]
