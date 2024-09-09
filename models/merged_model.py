# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 10:02:37 2023

merged CNN model for SLO and OCT images

@author: Roya Arian, royaarian101@gmail.com
"""

# Create two CNN models with the same fully connected layers
from tensorflow.keras.models import load_model
import tensorflow as tf
from models.SloModel import SloModel



def merged_model(input_img_slo ,input_img_oct):

    ### SLO model
    SloModel (input_img_slo)
    slo_model = load_model('slo_model_resnet101.h5')

    for layer in slo_model.layers:
        layer.trainable = False


    ### OCT model

    oct_model =  tf.keras.applications.resnet.ResNet101(
        weights='imagenet',
        include_top=False,
        input_shape=input_img_oct,
        )

    oct_model.get_layer(index = 0)._name = 'OCT'

    for layer in oct_model.layers:
        layer.trainable = False


    inputs_slo = tf.keras.layers.Input(input_img_slo)
    slo_output = tf.keras.layers.Flatten()(slo_model(inputs_slo))


    inputs_oct = tf.keras.layers.Input(input_img_oct)
    oct_output = tf.keras.layers.Flatten()(oct_model(inputs_oct))


    model = tf.keras.layers.Concatenate(axis=-1)([slo_output, oct_output])


    model = tf.keras.layers.Dropout(rate = 0.1)(model)

    model = tf.keras.layers.Dense(26, activation='relu')(model)

    model = tf.keras.layers.Dropout(rate = 0.1)(model)

    model = tf.keras.layers.Dense(13, activation='relu')(model)

    model = tf.keras.layers.Dropout(rate = 0.1)(model)

    model = tf.keras.layers.Dense(415, activation='relu')(model)

    model = tf.keras.layers.Dropout(rate = 0.0)(model)

    model = tf.keras.layers.Dense(2997, activation='relu')(model)

    model = tf.keras.layers.Dropout(rate = 0.3)(model)

    outputs = tf.keras.layers.Dense(1, 'sigmoid')(model)


    model_merged = tf.keras.Model([inputs_slo, inputs_oct] , outputs)

    return model_merged
