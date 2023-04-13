from Labeling import labeling
from Segmentation import file_processing
from FeatureExtraction import extraction_function
from FileMerging import file_merging
from DataShuffling import data_shuffler
from FeatureSelection import feature_elimination_loop, feature_selection
from RFC import rfc
from XGBoost import xgboost
from SVM import svm
from CVrfc import cvrfc
from CVxgboost import cvxgboost
from CVsvm import cvsvm

MainPath = "C:/Users/janja/OneDrive/Pulpit/DaneMGR"
extracted_dir = './CreatedFiles/Extracted_Features/'
label_dir = './CreatedFiles/Labels/'
merged_dir = './CreatedFiles/Entire_Set/'
shuffled_dir = './CreatedFiles/Shuffled_Data/'
segmented_dir = './CreatedFiles/Segmentation/'
feature_names = ['mean', 'mean_2', 'median', 'median_2', 'std', 'std_2', 'min_val', 'min_val_2',\
        'max_val', 'max_val_2', 'sum_val', 'sum_val_2', 'kurtosis', 'kurtosis_2', 'skew', 'skew_2', 'fft_sum', 'fft_sum_2', 'argmax_freq', 'argmax_freq_2']
feature_description_folder = './CreatedFiles/Feature_Description/features.txt' #descriptions of each number of features
selected_feature_folder = './CreatedFiles/Selected_Features/'
WindowSize = 500    #Windows size for segmentation
NumOfSensors = 2 #number of sensor
num_features = 10 #number of features for each sensor
num_of_folds = 5 #number of folds for cross validation


further_action = True
further_action = input("Do you want to continue? Please write 'yes' to confirm. ").lower() == 'yes'
while further_action == True:
    if further_action == False:
        break
    else:
        print('The steps possible to be completed are: \n \
        segmentation \n \
        feature_extraction \n \
        labeling \n \
        merging \n \
        shuffling \n \
        feature_loop \n \
        feature_selection \n \
        classification \n \
        cv_classification \n')
        step = input("Which step of the process do you want to use. Write the step with the '_' symbol instead of spaces         \n").lower()
        
        if (step == 'segmentation'):
            T = 'N' #500000 #'N' # describes which rows multiplied by n should be taken into the dataset #If chosen parameter is N the rows will not get dropped
            file_processing(T, WindowSize, NumOfSensors, MainPath)
        if (step == 'feature_extraction'):
            extraction_function(segmented_dir, num_features)
        if (step == 'labeling'): 
            labeling(extracted_dir)
        if (step == 'merging'):
            file_merging(num_features, NumOfSensors, extracted_dir, label_dir, merged_dir)
        if (step == 'shuffling'):
            data_shuffler(merged_dir, shuffled_dir)    
        if (step == 'feature_loop'):
            feature_elimination_loop(num_features, shuffled_dir, feature_names, feature_description_folder)
        if (step == 'feature_selection'):
            feature_selection(shuffled_dir, feature_names, selected_feature_folder)
        if (step == 'classification'):
            rfc()
            xgboost()
            svm()
        if (step == 'cv_classification'):
            cvrfc(num_of_folds) #random forest with cross validation
            cvxgboost(num_of_folds) #xgboost with cross validation
            cvsvm(num_of_folds) #svm with cross validation
            
        further_action = input("Do you want to continue? ").lower() == 'yes'