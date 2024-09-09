# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 13:12:34 2022

Calculating Metrics

@author: Roya Arian, royaarian101@gmail.com
"""

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
from sklearn import metrics
import sklearn


def metrics_calculation(y_valid, y_pred, y_prob):
    
    """
    A function for calculating the metrics 
    
    TP = confusion[1,1] # true positive 
    TN = confusion[0,0] # true negatives
    FP = confusion[0,1] # false positives
    FN = confusion[1,0] # false negatives
    cm = confusion matrix
    P_R_AUC: Precision-Recall AUC
    class_acc: acc of each class individually 
    """

    #####################################################
    #Get the confusion matrix
    #####################################################
    ROC_AUC = roc_auc_score(y_valid, y_prob)
    f1 = metrics.f1_score(y_valid, y_pred, average='weighted')
    precision, recall, thresholds = precision_recall_curve(y_valid, y_prob)
    P_R_AUC = auc(recall, precision)
    cm = sklearn.metrics.confusion_matrix(y_valid, y_pred, normalize='pred')
    #Now the normalize the diagonal entries
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    class_acc = cm.diagonal()

    Specificity = cm[0,0]/(cm[0,0]+cm[0,1])
    Sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])
    Precision   = cm[1,1]/(cm[0,1]+cm[1,1])


    return Specificity, Sensitivity, Precision, f1, ROC_AUC, P_R_AUC, class_acc, cm