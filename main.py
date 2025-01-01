##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                           Hunan University                           ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
from run_code.MSFA_CAVE     import main_MSFA_CAVE
from run_code.MSFA_RealData import main_MSFA_Realdata



if __name__ == '__main__':
    MSFA          = 1          # 1 for MSFA  task
    Real_exp      = 0          # 1 for Real MSFA exp.
    
    
    if MSFA:
        # 'CAVE_balloons' (3), 'CAVE_beads' (0.5), 'CAVE_cd' (0.1), 'CAVE_chart' (1), 'CAVE_clay' (1), 'CAVE_cloth' (5), 'CAVE_fake_bear' (1), 'CAVE_feathers' (1), 'CAVE_flowers' (3), 'CAVE_oil' （7）
        data_list = ['CAVE_oil'] 
        for file_name in data_list:
            main_MSFA_CAVE(file_name)
    elif Real_exp:
        data_list = ['real_data_25']
        for file_name in data_list:
            main_MSFA_Realdata(file_name)
