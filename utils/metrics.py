import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_curve,\
                    recall_score, classification_report, confusion_matrix, auc,\
                    precision_recall_curve, cohen_kappa_score
import numpy as np

def my_cal_roc(df):
    """
    Calculate the array of fpr and tpr, and the value of auroc

    Keyword arguments:
        df: n*2 dataframe
        each row represents a sample
        the first column Label is its true label
        the second Pred is the estimated probability

    Returns
    -------
    fpr: array
        array of false positive rate
    tpr: array
        array of true positive rate
    thresholds: array
        for analysis with bootstrap, threshold is set to an array from 0 to 1 with step 0.001
    auroc: numeric
        the area under roc
    """
    
    fpr, tpr, thresholds = roc_curve(df['Label'], df['Pred'])
    auroc = auc(fpr,tpr)
    return fpr, tpr, thresholds, auroc

def best_cutoff(y_predprob, y_test,is_plot=False):
    '''
    The optimal cut off has two criteria
    1. argmax(tpr-fpr), Youden index 
    2. argmin(abs(tpr - (1-fpr)))
    '''
    fpr, tpr, thresholds = roc_curve(y_test, y_predprob)
    roc_auc = auc(fpr, tpr)
#    print("Area under the ROC curve : %f" % roc_auc)
    

    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),
                    'tpr' : pd.Series(tpr, index = i), 
                    'tpr-fpr' : pd.Series(tpr-fpr, index=i),
                    '1-fpr' : pd.Series(1-fpr, index = i), 
                    'tf' : pd.Series(tpr - (1-fpr), index = i), 
                    'thresholds' : pd.Series(thresholds, index = i)})
    roc['abs_tf'] = np.abs(roc['tf'])
    roc = roc.sort_values('abs_tf')
    
    cutoff1 = np.argmax(roc['tpr-fpr'])
    cutoff2 = np.argmin(np.abs(roc['tpr'] - roc['1-fpr']))
    if is_plot:
        # Plot tpr vs 1-fpr
        fig, ax = plt.subplots()
        plt.plot(roc['thresholds'], roc['tpr'], label='tpr')
        plt.plot(roc['thresholds'], roc['1-fpr'], color = 'red', label='1-fpr')
        plt.plot(roc['thresholds'], roc['tpr-fpr'], color='g', label='tpr-fpr')
        plt.grid(color='silver')
        plt.xlabel('thresholds')
        #plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend()
        plt.text(roc.loc[cutoff1, 'thresholds'], roc.loc[cutoff1, 'tpr-fpr'], 'best cutoff st max(tpr-fpr): %4.2f'%(roc.loc[cutoff1, 'thresholds']))
        plt.plot(roc.loc[cutoff1, 'thresholds'], roc.loc[cutoff1, 'tpr-fpr'], 'bo')
        plt.text(roc.loc[cutoff2, 'thresholds'], roc.loc[cutoff2, 'tpr'], 'best cutoff st min(abs(sensi-speci)): %4.2f'%(roc.loc[cutoff2, 'thresholds']))
        plt.plot(roc.loc[cutoff2, 'thresholds'], roc.loc[cutoff2, 'tpr'], 'ro')

        plt.show()
    return roc.loc[cutoff1, 'thresholds'], roc.loc[cutoff2, 'thresholds']

def val_model_binary(y_test, y_proba,is_plot=False):
    best, _ = best_cutoff(y_proba, y_test)
    y_pred = y_proba>best

    cm = confusion_matrix(y_test, y_pred)
    # spec = cm[0,0] / (cm[0,0]+cm[0,1])
    # sens = cm[1,1] / (cm[1,0]+cm[1,1])

    fpr,tpr,thresholds = roc_curve(y_test,y_proba)
    auroc = auc(fpr, tpr)
    print('auroc is %4.4f' % auroc)

    if is_plot:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (auc = %0.2f)' % auroc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    else:
        pass
