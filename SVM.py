from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm


def svm():
    
    classification_folder = './CreatedFiles/Without_Cross_Validation/'
    X_train = np.load('./CreatedFiles/Selected_Features/X_train.npy')
    X_test = np.load('./CreatedFiles/Selected_Features/X_test.npy')
    y_train = np.load('./CreatedFiles/Selected_Features/y_train.npy')
    y_test = np.load('./CreatedFiles/Selected_Features/y_test.npy')
    # create the model
    clf = svm.SVC(kernel='rbf', C=1, gamma='scale')

    # fit the model to the training data
    clf.fit(X_train, y_train)

    # make predictions on the test set
    y_pred = clf.predict(X_test)

    # evaluate the model's performance
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (acc * 100.0))
    precision = precision_score(y_test, y_pred)
    print("Precision: %.2f%%" % (precision * 100.0))
    recall = recall_score(y_test, y_pred)
    print("Recall: %.2f%%" % (recall * 100.0))

    with open(f'{classification_folder}SVM.txt', 'w') as f:
        f.write('SVM without cross validation \
            Accuracy: %.2f%% \
                Precision: %.2f%% \
                    Recall: %.2f%%' % (acc * 100.0, precision * 100.0, recall * 100.0))
        
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

    cm_display.plot()
    
    plt.savefig(f'{classification_folder}SVM_matrix.png')
    plt.show()