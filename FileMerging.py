import os
import numpy as np
import re
#data_dir = './CreatedFiles/Extracted_Features/'
#label_dir = './CreatedFiles/Labels/'
#target_dir = './CreatedFiles/Entire_Set/'

def file_merging(num_features, num_sensors, data_dir, label_dir, target_dir):

    
    X = np.empty((0, num_features * num_sensors))
    y = np.empty(0)
    
    for filename in os.listdir(data_dir):
        file_loaded = np.load(data_dir + filename)
        
        X = np.concatenate((X, file_loaded), axis = 0)


        if sum(sum(np.isnan(file_loaded))) > 0:
            print(filename)
            print(file_loaded.shape)
            print(sum(sum(np.isnan(file_loaded))))
            break


    for label in os.listdir(label_dir):
        label_loaded = np.load(label_dir + label)

        y = np.concatenate((y, label_loaded), axis = 0)
    
    Data = f'{target_dir}X'
    Labels = f'{target_dir}y'
    np.save(Data, X)
    np.save(Labels, y)

