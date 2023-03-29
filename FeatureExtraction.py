import numpy as np
import scipy
from scipy import stats
from scipy.integrate import trapz
from scipy.fft import fft
import os


global num_features
num_features = 10
def feature_extraction(data):
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


        fft_sums = np.zeros((len(data),2), dtype = float)

        fft_freqs = np.zeros((len(data),2), dtype = float) ###Czy tu aby na pewno powinien być dloat, a nie liczba urojona

        #Flattening the array to shape (a*b, c), where c is the number of sensors
        flat_arr = data.reshape(len(data)*len(data[1]), len(data[0][1]))

        
        k = 0
        for k in range(len(data)):
                
                chunk = flat_arr[k * len(data[1]) : (k + 1) * len(data[1]), ]
                
                sensor_1 = chunk[:,0]
                sensor_2 = chunk[:,1]
                
                
                sp1 = np.fft.fft(sensor_1)
                ps1 = np.abs(sp1)**2
                sp2 = np.fft.fft(sensor_2)
                ps2 = np.abs(sp2)**2
                
                
                # Define the frequency range of interest
                sensor_1_freq = np.fft.fftfreq(len(chunk), 1/freq)
                
                idx1 = np.logical_and(sensor_1_freq >= 0, sensor_1_freq <= freq)

                # Integrate the power spectrum over the frequency range of interest
                area1 = trapz(ps1[idx1], sensor_1_freq[idx1])


                # Define the frequency range of interest
                sensor_2_freq = np.fft.fftfreq(len(chunk), 1/freq)
                
                idx2 = np.logical_and(sensor_2_freq >= 0, sensor_2_freq <= freq)

                # Integrate the power spectrum over the frequency range of interest
                area2 = trapz(ps2[idx1], sensor_2_freq[idx1])


                fft_sums[k] = [area1, area2]
                
                
                argmax_ind_1 = np.arange(len(sp1))
                argmax_list_1 = argmax_ind_1[np.argsort(-np.abs(sp1))]
                
                max_power_frequency_1 = argmax_list_1[0] * (freq / len(chunk))
                
                
                argmax_ind_2 = np.arange(len(sp2))
                argmax_list_2 = argmax_ind_2[np.argsort(-np.abs(sp2))]
                
                
                array_len_1 = len(argmax_list_1)
                
                for ind_1 in range(array_len_1):
                        
                        if (argmax_list_1[ind_1] * (freq / len(chunk)) <= 1):
                                
                                continue
                        
                        elif (argmax_list_1[ind_1] * (freq / len(chunk)) > 1):
                                max_power_frequency_1 = argmax_list_1[ind_1] * (freq / len(chunk))

                                break
                        
                array_len_2 = len(argmax_list_2)

                for ind_2 in range(array_len_2):
                        
                        if (argmax_list_2[ind_2] * (freq / len(chunk)) <= 1):
                                
                                continue
                        
                        elif (argmax_list_2[ind_2] * (freq / len(chunk)) > 1):
                                max_power_frequency_2 = argmax_list_2[ind_2] * (freq / len(chunk))
        
                                break
           
        
        
                fft_freqs[k] = [max_power_frequency_1, max_power_frequency_2]


        global feature_names
        feature_names = ['mean', 'mean_2', 'median', 'median_2', 'std', 'std_2', 'min_val', 'min_val_2',\
                'max_val', 'max_val_2', 'sum_val', 'sum_val_2', 'kurtosis', 'kurtosis_2', 'skew', 'skew_2', 'fft_sum', 'fft_sum_2', 'argmax_freq', 'argmax_freq_2']
        
        features = np.zeros((len(data), num_features))        

        features = np.concatenate((mean_array, median_array, std_array, min_val_array, max_val_array, sum_val_array, kurtosis_array, skewness_array, fft_sums, fft_freqs), axis = 1)
        
        return features
    
    
    

#segmented_dir = './CreatedFiles/Segmentation/'

def extraction_function(segmented_dir):
    for filename in os.listdir(segmented_dir):
        if filename.endswith('_B.npy'):
            file_loaded = np.load(segmented_dir + filename)
            extracted = feature_extraction(file_loaded)
            
            location_FE_B = f'./CreatedFiles/Extracted_Features/{filename}'
            np.save(location_FE_B, extracted)
        elif filename.endswith('_R.npy'):
            file_loaded = np.load(segmented_dir + filename)
            extracted = feature_extraction(file_loaded)
            
            location_FE_R = f'./CreatedFiles/Extracted_Features/{filename}'
            np.save(location_FE_R, extracted)
            continue
        else:
            print('Error file not found')
            break
