'''
SUMMARY:  config file
AUTHOR:   Aditya Arora
Created:  2018.03.09
Modified: 2018.03.09
--------------------------------------
'''


import numpy as np
import config as cfg
import sys
import cPickle
import os
import csv
import keras
import models as M
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.preprocessing import label_binarize

from keras.models import Model,Sequential
from keras.layers import Dense,Dropout
from keras import optimizers
from keras.layers import Convolution1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten
from keras import initializers, regularizers, constraints 
from keras.layers import Input, Merge,Lambda, Embedding, Bidirectional, LSTM, Dense, RepeatVector
from keras.layers import BatchNormalization


def mat_2d_to_3d(X, agg_num, hop):
    # pad to at least one block
    len_X, n_in = X.shape
    if (len_X < agg_num):
        X = np.concatenate((X, np.zeros((agg_num-len_X, n_in))))
        
    len_X = len(X)
    i1 = 0
    X3d = []
    while (i1+agg_num <= len_X):
        X3d.append(X[i1:i1+agg_num])
        i1 += hop
    return np.array(X3d)

train_x=np.load('melx.npy')
train_y=np.load('mely.npy')

[batch_num, n_time, n_freq] = train_x.shape
train_x=train_x.reshape((batch_num,1,n_time,n_freq))
    
input_neurons=200
dropout1=0.1
act1='relu'
act2='relu'
act3='sigmoid'
epochs=20
batchsize=100
agg_num=10
hop=10
dimx = n_time
dimy = n_freq
nb_filter = 100
filter_length =3
pool_size=(2,2)
labels = [ 'c', 'm', 'f', 'v', 'p', 'b', 'o', 'S' ]
num_classes=len(labels)

inpx = Input(shape=(1,dimx,dimy),name='inpx')

x = Conv2D(filters=nb_filter,
           kernel_size=filter_length,
           data_format='channels_first',
           padding='same',
           activation=act1)(inpx)

hx = MaxPooling2D(pool_size=pool_size)(x)
h = Flatten()(hx)
wrap = Dense(input_neurons, activation=act2,name='wrap')(h)
score = Dense(num_classes,activation=act3,name='score')(wrap)

model = Model([inpx],score)
model.compile(loss='mse',
          optimizer='adam',
          metrics=['mae'])

model.fit(train_x,train_y,batch_size=batchsize,epochs=epochs,verbose=1)