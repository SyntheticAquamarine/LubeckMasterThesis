import numpy as np
import os
import glob
import pandas as pd
from os.path import join
import math

data_dir = 'C:/Users/janja/OneDrive/Pulpit/DaneMGR'
#segment_dir =  '/Users/falcolentzsch/Develope/ScreenFM/SFM/data'
child_folder_paths = [f for f in os.listdir(data_dir)]

window_size = 100

for path in child_folder_paths:
    tmp_sub_path = join(data_dir, path)
    
    tmp_sub_B_path = join(tmp_sub_path, 'B')
    tmp_list_file_paths_B = [join(tmp_sub_B_path, f) for f in os.listdir(tmp_sub_B_path) if f.endswith('fast_Unknown.csv')]
    file_path_B = tmp_list_file_paths_B[0]
    df_B = pd.read_csv(file_path_B, sep=',', header=None)
    
    np_data = df_B.to_numpy()
    nb_timestamps, nb_sensors = np_data.shape
    
    #window_size = 100 # Size of the data segments, earlier there was the value of 100
    timestamp_idx = 0 # Index along the timestamp dimension
    segment_idx = 0 # Index for the segment dimension
    
    nb_segments = int(math.floor(nb_timestamps/window_size))
    print(f'Starting segmentation with a window size of {window_size} resulting in {nb_segments} segments.')
    data_to_save = np.zeros((nb_segments,window_size,nb_sensors),dtype=np.float32)

    while segment_idx < nb_segments:
        data_to_save[segment_idx] = np_data[timestamp_idx:timestamp_idx+window_size,:]
        timestamp_idx += window_size
        segment_idx += 1
        
        
    
    tmp_sub_S_path = join(tmp_sub_path, 'S')
    tmp_list_file_paths_S = [join(tmp_sub_S_path, f) for f in os.listdir(tmp_sub_S_path) if f.endswith('fast_Unknown.csv')]
    file_pat_S = tmp_list_file_paths_S[0]
    df_S = pd.read_csv(file_pat_S, sep=',', header=None)
    
    np_data = df_S.to_numpy()
    nb_timestamps, nb_sensors = np_data.shape
    
    #window_size = 100 # Size of the data segments, earlier there was the value of 100
    timestamp_idx = 0 # Index along the timestamp dimension
    segment_idx = 0 # Index for the segment dimension
    
    nb_segments = int(math.floor(nb_timestamps/window_size))
    print(f'Starting segmentation with a window size of {window_size} resulting in {nb_segments} segments.')
    data_to_save_S = np.zeros((nb_segments,window_size,nb_sensors),dtype=np.float32)

    while segment_idx < nb_segments:
        data_to_save[segment_idx] = np_data[timestamp_idx:timestamp_idx+window_size,:]
        timestamp_idx += window_size
        segment_idx += 1


    print('test')