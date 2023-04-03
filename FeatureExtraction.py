import numpy as np
import scipy
from scipy import stats
from scipy.integrate import trapz
from scipy.fft import fft
import os


#global num_features
#num_features = 10
def feature_extraction(data, num_features):
        freq = 100
        mean_array = np.zeros((len(data),2), dtype = float)
        median_array = np.zeros((len(data),2), dtype = float)
        std_array = np.zeros((len(data),2), dtype = float)
        min_val_array = np.zeros((len(data),2), dtype = float)
        max_val_array = np.zeros((len(data),2), dtype = float)
        sum_val_array = np.zeros((len(data),2), dtype = float)
        kurtosis_array = np.zeros((len(data),2), dtype = float)
        skewness_array = np.zeros((len(data),2), dtype = float)
        for i in range(len(data)):
                mean_array[i] = np.mean(data[i], axis=0)
                median_array[i] = np.median(data[i], axis=0)
                std_array[i] = np.std(data[i], axis=0)
                min_val_array[i] = np.min(data[i], axis=0)
                max_val_array[i] = np.max(data[i], axis=0)
                sum_val_array[i] = np.sum(data[i], axis=0)/freq
                kurtosis_array[i] = scipy.stats.kurtosis(data[i], axis=0)       
                skewness_array[i] = scipy.stats.skew(data[i], axis=0)


        fft_sums = np.zeros((len(data), data.shape[2]), dtype = float)
        fft_freqs = np.zeros((len(data), data.shape[2]), dtype = float)
        
        for depth in range(data.shape[0]):
                for sensor in range(data.shape[2]):

                        sp = np.fft.fft(data[depth, : , sensor])
                        ps = np.abs(sp)**2
                        
                        #calculating the area under the curve
                        data_freq = np.fft.fftfreq(data.shape[1], 1/freq)
                        #end_freq = np.array([frqc for frqc in data_freq if frqc > 0])
                        idx = np.logical_and(data_freq >= 0, data_freq <= freq)
                        area = trapz(ps[idx], data_freq[idx])
                        fft_sums[depth, sensor] = area
                        
                        #calculation of the greatest frequencies
                        argmax_ind = np.arange(len(sp))
                        argmax_list = argmax_ind[np.argsort(-ps)]
                        max_power_frequency = argmax_list[0] * (freq / data.shape[1])
                
                        array_len = len(argmax_list)
                
                        for ind in range(array_len):
                        
                                if (argmax_list[ind] * (freq / data.shape[1]) <= 1):
                                        
                                        continue
                                
                                elif (argmax_list[ind] * (freq / data.shape[1]) > 1):
                                        max_power_frequency = argmax_list[ind] * (freq / data.shape[1])

                                        break                        
                        fft_freqs[depth, sensor] = max_power_frequency


        global feature_names
        feature_names = ['mean', 'mean_2', 'median', 'median_2', 'std', 'std_2', 'min_val', 'min_val_2',\
                'max_val', 'max_val_2', 'sum_val', 'sum_val_2', 'kurtosis', 'kurtosis_2', 'skew', 'skew_2', 'fft_sum', 'fft_sum_2', 'argmax_freq', 'argmax_freq_2']
        
        features = np.zeros((len(data), num_features))        

        features = np.concatenate((mean_array, median_array, std_array, min_val_array, max_val_array, sum_val_array, kurtosis_array, skewness_array, fft_sums, fft_freqs), axis = 1)
        
        return features
    
    
    

#segmented_dir = './CreatedFiles/Segmentation/'

def extraction_function(segmented_dir, num_features):
    for filename in os.listdir(segmented_dir):
        if filename.endswith('_B.npy'):
            file_loaded = np.load(segmented_dir + filename)
            extracted = feature_extraction(file_loaded, num_features)
            
            location_FE_B = f'./CreatedFiles/Extracted_Features/{filename}'
            np.save(location_FE_B, extracted)
        elif filename.endswith('_R.npy'):
            file_loaded = np.load(segmented_dir + filename)
            extracted = feature_extraction(file_loaded, num_features)
            
            location_FE_R = f'./CreatedFiles/Extracted_Features/{filename}'
            np.save(location_FE_R, extracted)
            continue
        else:
            print('Error file not found')
            break
