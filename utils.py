# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 09:31:37 2023


Utilized functions

@author:Roya Arian, royaarian101@gmail.com
"""


import numpy as np
import matplotlib.pyplot as plt

def Initialize(number_class, nfold):
    
    """
    This function Initializes some evaluation parameters for the bi-modal classifier
    
    acc: Accuracy
    sp: Spesificity
    se: Sensitivity 
    pr: precision
    f1: f1-score
    auc: ROC AUC (AUROC)
    pr_auc: Precision-Recall AUC
    class_acc: acc of each class individually 
    
    """
    
    d = dict()
    d['nfold'] = nfold
    d['target_names'] = ['Normal' , 'MS']  # classes based on lables (Normal = 0, MS = 1)
    
    
    ### Initializing
    
    d['cnn_acc']   = []  # acc of validation dataset
    d['test_acc']  = []  # acc of test dataset


    d['cnn_se']   = np.zeros((nfold)) # se of validation dataset
    d['test_se']  = np.zeros((nfold)) # se of test dataset


    d['cnn_sp']   = np.zeros((nfold)) # sp of validation dataset
    d['test_sp']  = np.zeros((nfold)) # sp of test dataset
    

    d['cnn_pr']   = np.zeros((nfold)) # pr of validation dataset
    d['test_pr']  = np.zeros((nfold)) # pr of test dataset


    d['cnn_f1']   = np.zeros((nfold)) # f1 of validation dataset
    d['test_f1']  = np.zeros((nfold)) # f1 of test dataset


    d['cnn_auc']   = np.zeros((nfold)) # auc of validation dataset
    d['test_auc']  = np.zeros((nfold)) # auc of test dataset


    d['cnn_pr_auc']   = np.zeros((nfold)) # pr_auc of validation dataset
    d['test_pr_auc']  = np.zeros((nfold)) # pr_auc of test dataset


    d['class_acc']       = np.zeros((number_class)) # class_acc of validation dataset
    d['class_acc_test']  = np.zeros((number_class)) # class_acc of test dataset

    
    d['confusion_matrix']      = np.zeros((number_class, number_class)) # confusion_matrix of validation dataset
    d['confusion_matrix_test'] = np.zeros((number_class, number_class)) # confusion_matrix of test dataset
      
    return d




def printing(d):
    
    """
   This function prints some evaluation parameters for the bi-modal classifier
    
    acc: Accuracy
    sp: Spesificity
    se: Sensitivity 
    pr: precision
    f1: f1-score
    auc: ROC AUC (AUROC)
    pr_auc: Precision-Recall AUC
    class_acc: acc of each class individually 
    """
    
    print('Validation acc     = %f' % np.mean(d['cnn_acc']))   
    print('Validation se      = %f' % np.mean(d['cnn_se']))
    print('Validation sp      = %f' % np.mean(d['cnn_sp']))
    print('Validation pr      = %f' % np.mean(d['cnn_pr']))
    print('Validation f1      = %f' % np.mean(d['cnn_f1']))
    print('Validation AUROC   = %f' % np.mean(d['cnn_auc']))
    print('Validation PRAUC   = %f' % np.mean(d['cnn_pr_auc']))
    
    print('Validation acc of class %s' % d['target_names'][0], '= %f' % (d['cnn_acc'][0]/d['nfold']))
    print('Validation acc of class %s' % d['target_names'][1], '= %f' % (d['cnn_acc'][1]/d['nfold']), end='\n\n\n')
    
    
    
    
    print('test acc     = %f' % np.mean(d['test_acc']))   
    print('test se      = %f' % np.mean(d['test_se']))
    print('test sp      = %f' % np.mean(d['test_sp']))
    print('test pr      = %f' % np.mean(d['test_pr']))
    print('test f1      = %f' % np.mean(d['test_f1']))
    print('test AUROC   = %f' % np.mean(d['test_auc']))
    print('test PRAUC   = %f' % np.mean(d['test_pr_auc']))

    print('test acc of class %s' % d['target_names'][0], '= %f' % (d['class_acc_test'][0]/d['nfold']))
    print('test acc of class %s' % d['target_names'][1], '= %f' % (d['class_acc_test'][1]/d['nfold']), end='\n\n\n')
    