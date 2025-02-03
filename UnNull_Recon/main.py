##############################################################################
####                             Yurong Chen                              ####
####                      chenyurong1998 AT outlook.com                   ####
####                           Hunan University                           ####
####                       Happy Coding Happy Ending                      ####
##############################################################################

import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
from run_code.MSFA_CAVE     import main_MSFA_CAVE



if __name__ == '__main__':
    # 'CAVE_balloons', 'CAVE_beads', 'CAVE_cd', 'CAVE_chart', 'CAVE_clay', 'CAVE_cloth', 'CAVE_fake_bear', 'CAVE_feathers', 'CAVE_flowers', 'CAVE_oil'
    data_list = ['CAVE_feathers'] 
    for file_name in data_list:
        main_MSFA_CAVE(file_name)