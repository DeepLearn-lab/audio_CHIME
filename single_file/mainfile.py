'''
SUMMARY:  main file
AUTHOR:   DL-LAB
Created:  2018.03.09
Modified: 2018.03.10
--------------------------------------
'''


import numpy as np
import keras

from keras.models import Model,Sequential
from keras.layers import Input, Merge, Dense, Dropout, Conv2D, MaxPooling2D, Flatten

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
model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['mae'])

model.fit(train_x,train_y,batch_size=batchsize,epochs=epochs,verbose=1)
