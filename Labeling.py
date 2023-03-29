import numpy as np
import re
import os
#data_dir = './CreatedFiles/Extracted_Features/'

def labeling(data_dir):
    for filename in os.listdir(data_dir):
        if filename.endswith('_B.npy'):
            
            file_loaded = np.load(data_dir + filename)
            labels = np.zeros(file_loaded.shape[0])
            label_file = re.sub('(\.npy)','',filename)
            
            location_FE_B = f'./CreatedFiles/Labels/{label_file}_label.npy'
            np.save(location_FE_B, labels)
            
        elif filename.endswith('_R.npy'):
            
            file_loaded = np.load(data_dir + filename)
            labels = np.zeros(file_loaded.shape[0])
            label_file = re.sub('(\.npy)','',filename)
            
            location_FE_R = f'./CreatedFiles/Labels/{label_file}_label.npy'
            np.save(location_FE_R, labels)
        else:
            print('Filename error')
            break