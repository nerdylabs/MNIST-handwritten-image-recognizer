#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:25:14 2020

@author: heisenberg
"""

import pandas as pd
import numpy as np
import tensorflow as tf

dataset = pd.read_csv('train digit.csv')

X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:,0].values

Y.shape

from sklearn import preprocessing
X = preprocessing.scale(X)

from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.2)

Ytrain.shape

Xtrain.shape
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


Ypred = ANN.predict(X)

h = []
for i in range(np.size(Ytest)):
    h.append(np.where(Ypred[i][:] == max(Ypred[i][:])))
    
df = pd.DataFrame(h)


yprediction = ANN.predict_classes(Xtest)

from sklearn.metrics import confusion_matrix, accuracy_score

cm  = confusion_matrix(Ytest, yprediction)
print(cm)
accuracy_score(Ytest, yprediction)


datatest = pd.read_csv('test.csv')
Xkaggle = datatest.iloc[:,:].values

Ykaggle4 = ANN.predict_classes(Xkaggle)

dfk4 = pd.DataFrame(Ykaggle4)
dfk4.to_csv('submission4.csv', header=False, index=True) 