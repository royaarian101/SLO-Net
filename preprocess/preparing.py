# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 11:59:32 2022

@author: Roya Arian, royaarian101@gmail.com
"""
import numpy as np
def preparing(x, y):
    """
    
      This function prepares the data when the input is a dictionary and the output is a NumPy array.
      The data is primarily a dictionary where each key represents one subject, as we want to split 
      the data based on subjects. However, for training the models, we need a NumPy array.
    
    """  

    data_slo  = []
    data_oct  = []
    label     = []
    for i in x:
      for j in range(len(x[i])):
          data_slo.append(np.array(x[i][j][0])/255)
          data_oct.append(np.array(x[i][j][1])*255)
          label.append(y[i])
    data_slo = np.reshape(data_slo, np.shape(data_slo))
    data_oct = np.reshape(data_oct, np.shape(data_oct))
    return data_slo, data_oct, np.array(label)