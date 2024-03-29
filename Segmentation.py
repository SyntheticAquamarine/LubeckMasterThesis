import os
import pandas as pd
from pandas import read_csv
import re
from os import walk
import numpy as np
import math


def file_import(MainPath):

    filepaths = []

    for root, dirs, files in os.walk(MainPath, topdown=True):
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



def file_processing(T, WindowSize, MainPath):
    file_import(MainPath)
    #T = 'N' #500000 #'N' # describes which rows multiplied by n should be taken into the dataset #If chosen parameter is N the rows will not get dropped
    fields = ['Time', 'Infinity|RESP.ONLY_ONE_IN_GROUP [OHM]', 'Infinity|SPO2.SPO2_PULSE [COUNTS]']
    #WindowSize = 100
    #NumOfSensors = 2

    files_total = len(fullPath)
    i = 1
    
    
    SensorPath = 'C:/Users/janja/LubeckMasterThesis/SensorFiles'

    directory = os.fsencode(SensorPath)
     
 
    for path in fullPath:
        print(path)
        
        df_local = read_csv(path, sep = ',', encoding = 'UTF-8', usecols=fields)
        df_local.rename(columns = {"Time":"timestamp"}, inplace = True)
        
        
        imported = re.sub('\/','',re.search('\/\d{1,2}\/[B,R]\/',path).group())
        print(f'The filename is: {imported}')
        
        sensor = read_csv(f'C:/Users/janja/LubeckMasterThesis/SensorFiles/{imported}.csv', sep = ';', header = 0)
        
        sensor = sensor[sensor.columns[1:]]
        sensor['timestamp'] = sensor['timestamp'].str.slice(start=11)
        sensor['timestamp'] = sensor['timestamp'].str.strip()

        df_local = df_local.interpolate()
        df_local.dropna(thresh = 3, inplace = True)
    
        
        df_local['timestamp'] = pd.to_datetime(df_local['timestamp'], format='%H:%M:%S.%f')
        df_local = df_local.set_index('timestamp')
        sensor['timestamp'] = pd.to_datetime(sensor['timestamp'], format='%H:%M:%S.%f')
        sensor = sensor.set_index('timestamp')
        print(f'Base frequency {sensor.index.freq}')
        sensor = sensor.resample('10ms').mean()
        print(f'Upsampled frequency {sensor.index.freq}')
        sensor = sensor.interpolate()

        
        print(f'sensor shape: {sensor.shape}')
        print(f' df_local shape {df_local.shape}')
        df_local = df_local.sort_values('timestamp')
        sensor = sensor.sort_values('timestamp')
        df_joined = pd.merge_asof(df_local, sensor, on = 'timestamp' , tolerance = pd.Timedelta('50ms'))
        df_joined = df_joined.drop(columns = ['timestamp'], axis = 1)
    
        
        print(f'na number pre dropna {df_joined.isna().sum()}')
        print(f'df joined shape {df_joined.shape}')
        df_joined.dropna(thresh = 8, inplace = True)
        print(f'na number post dropna {df_joined.isna().sum()}')
        print(f'df joined shape {df_joined.shape}')    
        print(df_joined)

    
    #print(f'Number of blank spaces for the position: \n {df_local.isna().sum()}')
        if T != 'N':
            df_joined = df_joined[df_joined.index % T == 0] #Set to 2000 as 1 second is 20 observations
        match = re.findall("\/B\/Csv",path)
        
    
        
        df_segmented = segmentation(df_joined, WindowSize)
        
        print('df_segmented shape')
        print(df_segmented.shape)
        print(sum(sum(np.isnan(df_segmented))))
        
                
        neo = re.findall('\/([\d]{1,2})\/', path)

        
        if bool(match) == True:
            location_B = f'./CreatedFiles/Segmentation/{neo[0]}_B.npy'
            np.save(location_B, df_segmented)

        else:
            location_R = f'./CreatedFiles/Segmentation/{neo[0]}_R.npy'
            np.save(location_R, df_segmented)

        print(f"Imported file number: {i}, from files total: {files_total}, and that is {i*100/files_total:.2f}%")
        i+=1 
    