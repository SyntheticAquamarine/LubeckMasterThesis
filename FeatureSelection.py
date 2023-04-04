from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder


def feature_elimination(i, shuffled_dir, feature_names):
    
    #for i in range(n*2,0,-1):
    
    # Create the Random Forest classifier
    rf = RandomForestClassifier()
    
    # Perform feature selection using RFE
    rfe = RFE(estimator=rf, n_features_to_select = i, step=1)
    
    
    X = np.load(f'{shuffled_dir}X_shuffled.npy')
    y = np.load(f'{shuffled_dir}y_shuffled.npy')
    
    rfe.fit(X, y)

    # Get the selected feature indices
    selected_features = rfe.support_
    selected_features_indices = np.where(selected_features)[0]
    print(selected_features_indices)
    print('Number of features selected: %d' % (i))
    
    names_selected_features = []


    for i in selected_features_indices:
        names_selected_features.append(feature_names[i])
    
    
    print('Features selected: ')
    print(names_selected_features)
    
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    # Use the selected features to train and evaluate the classifier
    global X_selected
    X_selected = X[:, selected_features]
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

    
    
    # Train the Random Forest classifier
    rf.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    mae = mean_absolute_error(y_test, y_pred)

    accuracy_stat = "Accuracy: %.4f%%" % (accuracy * 100.0)
    precision_stat = "Precision: %.4f%%" % (precision * 100.0)
    recall_stat = "Recall: %.4f%%" % (recall * 100.0)

    print("Accuracy: %.4f%%" % (accuracy * 100.0))
    print("Precision: %.4f%%" % (precision * 100.0))
    print("Recall: %.4f%%" % (recall * 100.0))
    print("Mean Absolute Error:", mae, end = '\n')
    
    return X_train, X_test, y_train, y_test, X_selected, X, y, accuracy_stat, precision_stat, recall_stat, names_selected_features
    
    
    
    
def feature_elimination_loop(num_features, shuffled_dir, feature_names, feature_description_folder):

    with open(feature_description_folder, 'w') as f:
        pass
    
    for i in range(num_features*2,0,-1):
        statistics = feature_elimination(i, shuffled_dir, feature_names)
        
        with open(feature_description_folder,'a') as f:
            #features = f'number_of_features: {i}'
            selected = statistics[4]
            accuracy = statistics[7]
            precision = statistics[8]
            recall = statistics[9]
            feature_names = statistics[10]
            f.write(f'Number of features:{i}, accuracy: {accuracy}, precision: {precision}, recall: {recall},\
                \n feature names {feature_names}, \n {selected} \n\n\n')

    
    #location_FE_R = f'./CreatedFiles/Labels/{label_file}_label.npy'
    #np.save(location_FE_R, labels)
    def feature_selection(shuffled_dir, feature_names, selected_feature_folder):
        Feature_no_chosen = int(input("How many features are to be selected?"))
        End_features = feature_elimination(Feature_no_chosen, shuffled_dir, feature_names)
        np.save(f'{selected_feature_folder}X_train.npy', End_features[0])
        np.save(f'{selected_feature_folder}X_test.npy', End_features[1])
        np.save(f'{selected_feature_folder}y_train.npy', End_features[2])
        np.save(f'{selected_feature_folder}y_test.npy', End_features[3])
        
    
    
    
    
    