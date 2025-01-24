import numpy as np
import seaborn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


def display_results(Y_test, predictions, predictions_proba):
    # collect metrics
    cm = confusion_matrix(Y_test, predictions, labels=[0, 1])
    accuracy = accuracy_score(Y_test, predictions)
    precision = precision_score(Y_test, predictions)
    recall = recall_score(Y_test, predictions)
    f1 = f1_score(Y_test, predictions)
    #flatten predictions if necessary (needed for built-in models to work)
    if predictions_proba.ndim > 1:
        predictions_proba = predictions_proba[:,1]
    #collect ROC AUC values
    fpr, tpr, threshold = roc_curve(Y_test, predictions_proba)
    roc_auc = auc(fpr, tpr)
    # display results
    print(f'Prediction results: ')
    print(f'Accuracy: {100 * accuracy:.4f} %')
    print(f'Precision: {100 * precision:.4f} %')
    print(f'Recall: {100 * recall:.4f} %')
    print(f'F1: {100 * f1:.4f} %')
    print(f'ROC_AUC: {roc_auc:.4f}')
    # confusion matrix
    hmap = seaborn.heatmap(cm, annot=True, fmt='g')
    # ROC AUC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Counterfeit Banknotes Detection')
    plt.legend()
    plt.show()


def save_custom(model, model_name):
    np.savetxt('saved_params/' + model_name + '_weights.txt', model.weights)
    np.savetxt('saved_params/' + model_name + '_bias.txt', np.array([model.bias]))


def load_custom(model_name):
    weights = np.loadtxt('saved_params/' + model_name + '_weights.txt')
    bias = np.loadtxt('saved_params/' + model_name + '_bias.txt')
    return weights, bias
