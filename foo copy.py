import numpy as np
import os
import glob
import pandas as pd
from os.path import join

data_dir = 'C:/Users/janja/OneDrive/Pulpit/DaneMGR'
segment_dir =  '/Users/falcolentzsch/Develope/ScreenFM/SFM/data'
child_folder_paths = [f for f in os.listdir(data_dir)]


for path in child_folder_paths:
    tmp_sub_path = join(data_dir, path)
    
    tmp_sub_B_path = join(tmp_sub_path, 'B')
    tmp_list_file_paths_B = [join(tmp_sub_B_path, f) for f in os.listdir(tmp_sub_B_path) if f.endswith('fast_Unknown.csv')]
    file_path_B = tmp_list_file_paths_B[0]
    df_B = pd.read_csv(file_path_B, sep=',', header=None)
    
    
    
    tmp_sub_S_path = join(tmp_sub_path, 'S')
    tmp_list_file_paths_S = [join(tmp_sub_S_path, f) for f in os.listdir(tmp_sub_S_path) if f.endswith('fast_Unknown.csv')]
    file_pat_S = tmp_list_file_paths_S[0]
    df_S = pd.read_csv(file_pat_S, sep=',', header=None)
    
    

    print('test')