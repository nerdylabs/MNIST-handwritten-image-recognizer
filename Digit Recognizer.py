#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:25:14 2020

@author: heisenberg
"""

import pandas as pd
import numpy as np
import tensorflow as tf

(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()
xtrain = xtrain.reshape(xtrain.shape[0], 28,28,1)
xtest = xtest.reshape(xtest.shape[0], 28,28,1)
xtrain =  xtrain.astype('float32')
xtest =  xtest.astype('float32')
xtrain /= 255
xtest /= 255
ytrain = tf.keras.utils.to_categorical(ytrain, 10)
ytest = tf.keras.utils.to_categorical(ytest, 10)
ANN = tf.keras.models.Sequential()

ANN.add(tf.keras.layers.Dense(784))
ANN.add(tf.keras.layers.Dense(50, activation= 'relu'))

ANN.add(tf.keras.layers.Dense(50, activation= 'relu'))
ANN.add(tf.keras.layers.Dropout(0.25))
ANN.add(tf.keras.layers.Dense(50, activation= 'relu'))
ANN.add(tf.keras.layers.Dropout(0.25))
ANN.add(tf.keras.layers.Dense(50, activation= 'relu'))
ANN.add(tf.keras.layers.Dropout(0.5))
ANN.add(tf.keras.layers.Dense(50, activation= 'relu'))

ANN.add(tf.keras.layers.Dense(10, activation= 'softmax'))

ANN.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

ANN.fit(Xtrain, Ytrain, batch_size=100, epochs=50, verbose=2)


yprediction = ANN.predict_classes(Xtest)

from sklearn.metrics import confusion_matrix, accuracy_score

cm  = confusion_matrix(Ytest, yprediction)
print(cm)
accuracy_score(Ytest, yprediction)

