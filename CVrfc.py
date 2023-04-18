from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def cvrfc(n_folds):

    classification_folder = './CreatedFiles/Classification/With_Cross_Validation/'
    X = np.load('./CreatedFiles/Selected_Features/X_selected.npy')
    y = np.load('./CreatedFiles/Selected_Features/y_selected.npy')
    
    # Create an instance of the KFold class
    kf = KFold(n_splits=n_folds, random_state=42, shuffle=True)

    # Define the RFC model
    rfc = RandomForestClassifier(n_jobs = -1)

    # Initialize a list to store the accuracy scores
    acc_scores_RFC = []
    precision_scores_RFC = []
    recall_scores_RFC = []

    # Perform the K-fold cross-validation
    for train_index, test_index in kf.split(X):
        # Split the data into train and test sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit the model to the training data
        rfc.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = rfc.predict(X_test)

        # Calculate the scores
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        acc_scores_RFC.append(acc)
        precision_scores_RFC.append(precision)
        recall_scores_RFC.append(recall)

    # Print the mean scores
    print('Random Forest with cross validation')
    print("Mean accuracy:", np.mean(acc_scores_RFC))
    print("Mean precision:", np.mean(precision_scores_RFC))
    print("Mean recall:", np.mean(recall_scores_RFC))


    with open(f'{classification_folder}CVrfc.txt', 'w') as f:
        f.write('Random Forest with cross validation \
            Accuracy: %.2f%% \
                Precision: %.2f%% \
                    Recall: %.2f%%' % (np.mean(acc_scores_RFC) * 100.0, np.mean(precision_scores_RFC) * 100.0, np.mean(recall_scores_RFC) * 100.0))

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

    cm_display.plot()
    
    plt.savefig(f'{classification_folder}CVrfc_matrix.png')
    plt.show()