# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 10:02:37 2023

Creating Resnet for SLO images

@author: Roya Arian, royaarian101@gmail.com
"""

### SLO model
import tensorflow as tf
from tensorflow.keras.models import Model

def SloModel (input_img_slo):
    slo_model =  tf.keras.applications.ResNet101(
        weights='imagenet',
        include_top=False,
        input_shape=input_img_slo,
        )
    
    for layer in slo_model.layers:
        layer._name = 'SLO_' + layer.name
    
    new_model = Model(inputs=slo_model.input, outputs=slo_model.output)
    
    new_model.save('slo_model_resnet101.h5')
