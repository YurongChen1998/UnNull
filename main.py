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
        # 'CAVE_balloons', 'CAVE_beads' (0.5), 'CAVE_cd', 'CAVE_chart', 'CAVE_clay', 'CAVE_cloth', 'CAVE_fake_bear', 'CAVE_feathers', 'CAVE_flowers', 'CAVE_oil'
        data_list = ['CAVE_balloons'] 
        for file_name in data_list:
            main_MSFA_CAVE(file_name)
    elif Real_exp:
        data_list = ['real_data_25']
        for file_name in data_list:
            main_MSFA_Realdata(file_name)
