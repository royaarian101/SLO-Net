# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 10:30:15 2024

@author: Roya Arian  email: royaarian101@gmail.com
"""

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold (n_splits = 5, shuffle = True, random_state = None)

import tensorflow as tf
import numpy as np
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import keras
import preprocess
import models
import metrics_losses
import utils

images_train = pickle.load(open("train_merged_data_with_sp.pkl", 'rb'))
labels_train = pickle.load(open("train_merged_data_with_sp_labels.pkl", 'rb'))

images_test = pickle.load(open("test_merged_data_with_sp.pkl", 'rb'))
labels_test = pickle.load(open("test_merged_data_with_sp_labels.pkl", 'rb'))

images_test_slo, images_test_oct, labels_test_slo = preprocess.preparing(images_test,labels_test)

images_test_slo = np.repeat (images_test_slo, repeats = 3, axis = 3)
images_test_oct = np.repeat (images_test_oct, repeats = 3, axis = 3)
#####################################################################
## Parameters
#####################################################################
channel = 1
number_class = 2  # please choose the number of classes according to your project
nfold = 5   # please choose number of folds in k fold cross validation algorithm

# initial some parameters
d = utils.Initialize(number_class, nfold)


#### model parameters
batch_size        = 16
epoch             = 100
learning_rate     = 0.000275167073
#####################################################################
## Applying kfold
#####################################################################

kf_nfold = StratifiedKFold(n_splits=nfold, random_state=42, shuffle=True)

n = 0
for train_index, val_index in kf_nfold.split(images_train,list(labels_train.values())):
    n = n+1
    # print(train_index, val_index)  # you can watch train and validation index using this comment
    print(f'---------------------------------------------------------------------\
          \n \t\t\t {n}th fold \n---------------------------------------------------------------------'\
          ,end = '\n\n\n' )
    x_train = {i: images_train[list(images_train.keys())[i]] for i in train_index}
    x_valid = {i: images_train[list(images_train.keys())[i]] for i in val_index}

    y_trainn = {i: labels_train[list(labels_train.keys())[i]] for i in train_index}
    y_validd = {i: labels_train[list(labels_train.keys())[i]] for i in val_index}


    ################## preparing

    x_train_slo, x_train_oct, y_train = preprocess.preparing(x_train,y_trainn)
    x_valid_slo, x_valid_oct, y_valid = preprocess.preparing(x_valid,y_validd)

    ################# Augmentation
    x_train_slo, y_train_slo = preprocess.Augmentation.Augmentation_slo(x_train_slo,y_train)

    x_train_oct, y_train_oct = preprocess.Augmentation.Augmentation_oct(x_train_oct,y_train)


    indices = np.random.permutation (len (x_train_slo))
    x_train_slo = x_train_slo [indices]
    y_train_slo = y_train_slo [indices]

    x_train_oct = x_train_oct [indices]
    y_train_oct = y_train_oct [indices]


    x_train_slo = np.repeat (x_train_slo, repeats = 3, axis = 3)

    x_train_oct = np.repeat (x_train_oct, repeats = 3, axis = 3)

    x_valid_slo = np.repeat (x_valid_slo, repeats = 3, axis = 3)

    x_valid_oct = np.repeat (x_valid_oct, repeats = 3, axis = 3)

    ####################################################################
    # classification
    ####################################################################

    input_img_slo = (np.shape(x_train_slo)[1], np.shape(x_train_slo)[2], 3)
    input_img_oct = (np.shape(x_train_oct)[1], np.shape(x_train_oct)[2], 3)

    model = models.merged_model(input_img_slo=input_img_slo, input_img_oct=input_img_oct)

    METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.AUC(name='auc'),
      ]


    my_optimizer =  tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=my_optimizer, loss="binary_crossentropy", metrics=METRICS)
    callbacks = [EarlyStopping(patience=20, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-6),
        ModelCheckpoint(f'slo_oct{n}.h5', verbose=1, save_best_only=True, save_weights_only=True)]

    #################################
    ###### Applying model  ###########
    #################################


    results = model.fit([x_train_slo, x_train_oct], y_train_slo, batch_size=batch_size, epochs=epoch, callbacks=callbacks,\
                    validation_data=([x_valid_slo, x_valid_oct], np.asarray(y_valid, dtype=np.float64)))


    plt.figure(figsize=(5, 5))
    plt.title(f"Learning curve {n}th fold")
    plt.plot(results.history["loss"][:-7], label="loss")
    plt.plot(results.history["val_loss"][:-7], label="val_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()

    plt.figure(figsize=(5, 5))
    plt.title(f"Learning curve {n}th fold")
    plt.plot(results.history["accuracy"], label="accuracy")
    plt.plot(results.history["val_accuracy"], label="val_accuracy")
    plt.plot( np.argmax(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]),\
             marker="x", color="r", label="best accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    plt.legend()


    # load the best model
    model.load_weights(f'slo_oct{n}.h5')


    pred_proba = model.predict([x_valid_slo, x_valid_oct]).ravel()
    pred_class = (pred_proba > 0.5).astype(np.uint8)


    ##### calculating metrics

    d['cnn_acc'].append(metrics.accuracy_score(y_valid, pred_class))
    print(f'accuracy of {n}th fold : {metrics.accuracy_score(y_valid, pred_class)}')
    d['cnn_sp'][n-1], d['cnn_se'][n-1], d['cnn_pr'][n-1], d['cnn_f1'][n-1], d['cnn_auc'][n-1]\
        , d['cnn_pr_auc'][n-1], acc_class, cm \
        = metrics_losses.metrics_calculation(y_valid, pred_class, pred_proba)

    #################### acc for each class ##################
    d['class_acc'] = np.add(d['class_acc'],acc_class)

    ###################### Total confusion_matrix for poly kernel ############
    d['confusion_matrix'] = np.add(d['confusion_matrix'],cm)

######################## internal test
    pred_proba_test = model.predict([images_test_slo, images_test_oct]).ravel()
    pred_class_test = (pred_proba_test > 0.5).astype(np.uint8)


    ##### calculating metrics

    d['test_acc'].append(metrics.accuracy_score(labels_test_slo, pred_class_test))
    print(f'test accuracy of {n}th fold : {metrics.accuracy_score(labels_test_slo, pred_class_test)}')
    d['test_sp'][n-1], d['test_se'][n-1], d['test_pr'][n-1], d['test_f1'][n-1], d['test_auc'][n-1]\
        , d['test_pr_auc'][n-1], acc_class, cm \
        = metrics_losses.metrics_calculation(y_valid, pred_class, pred_proba)

    #################### acc for each class ##################
    d['class_acc_test'] = np.add(d['class_acc_test'],acc_class)

    ###################### Total confusion_matrix for poly kernel ############
    d['confusion_matrix_test'] = np.add(d['confusion_matrix_test'],cm)
    

########################################
#     Metrics printing
########################################
utils.printing(d)

########################################
#     ploting confusion matrix
########################################
disp = ConfusionMatrixDisplay(confusion_matrix=d['confusion_matrix']//d['nfold'], display_labels=d['target_names'])
disp.plot()
