import numpy as np
from sklearn.preprocessing import LabelEncoder

def data_shuffler(merged_dir, shuffled_data):
    
    X_to_shuffle = f'{merged_dir}X.npy'
    y_to_shuffle = f'{merged_dir}y.npy'
    
    
    dataset = np.load(X_to_shuffle)
    labels = np.load(y_to_shuffle)
    shuffler = np.random.permutation(len(dataset))
    X = dataset[shuffler]
    y = labels[shuffler]
    
    np.save(f'{shuffled_data}/X_shuffled', X)
    np.save(f'{shuffled_data}/y_shuffled', y)


    print(sum(sum(np.isnan(X))))
    print(sum(sum(np.isnan(y))))
    return X,y
