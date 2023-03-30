import os
import pandas as pd
import re
from os import walk
import numpy as np
import math


def file_import():

    filepaths = []

    for root, dirs, files in os.walk("C:/Users/janja/OneDrive/Pulpit/DaneMGR", topdown=True):
        for name in dirs:
            if (bool(re.findall('\d$', name)) == False):
                Path = (root + '/' + name)
                filepaths.append(re.sub('DaneMGR\\\\', 'DaneMGR/', Path))
                

    filenameList = []
    global fullPath
    fullPath = []
    for Path in filepaths:
        for (dirpath, dirnames, filenames) in walk(Path):
            for name in filenames:
                if (bool(re.findall('fast_Unknown', name)) == True) and name not in filenameList:
                    NewName = re.sub('._CsvLog', 'CsvLog', name)
                    filenameList.append(NewName)
                    fullPath.append(Path + '/' + NewName)
                   
                   
                   
def segmentation(pd_data, window_size):
    np_data = pd_data.to_numpy()
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
    return data_to_save



def file_processing(T, WindowSize, NumOfSensors):
    file_import()
    #T = 'N' #500000 #'N' # describes which rows multiplied by n should be taken into the dataset #If chosen parameter is N the rows will not get dropped
    fields = ['Infinity|RESP.ONLY_ONE_IN_GROUP [OHM]', 'Infinity|SPO2.SPO2_PULSE [COUNTS]']
    #WindowSize = 100
    #NumOfSensors = 2
    df_0 = np.zeros((1,WindowSize, NumOfSensors))
    df_1 = np.zeros((1,WindowSize, NumOfSensors))

    files_total = len(fullPath)
    i = 1
    for path in fullPath:
        #print(path)
        df_local = pd.read_csv(path, sep = ',', encoding = 'UTF-8', usecols=fields)
        
        if re.findall('(\/23\/B)', path):
            print('23B') #41 row (41 row correct value)
        elif re.findall('(\/43\/R)', path):
            print('23B')    #81 (81 row)

        
        
        df_local = df_local.interpolate()
        if T != 'N':
            df_local = df_local[df_local.index % T == 0] #Set to 2000 as 1 second is 20 observations
        match = re.findall("\/B\/Csv",path)
        
        df_segmented = segmentation(df_local, WindowSize)
        
        #print(df_segmented)
        #print(df_segmented.shape)
        
        neo = re.findall('\/([\d]{1,2})\/', path)
        

        if bool(match) == True:
            location_B = f'./CreatedFiles/Segmentation/{neo[0]}_B.npy'
            #np.save(location_B, df_segmented)

        else:
            location_R = f'./CreatedFiles/Segmentation/{neo[0]}_R.npy'
            #np.save(location_R, df_segmented)

        print(f"Imported file number: {i}, from files total: {files_total}, and that is {i*100/files_total:.2f}%")
        i+=1 
    