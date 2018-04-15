# -*- coding: utf-8 -*-
"""
** deeplean-ai.com **
** dl-lab **
created by :: adityac8
"""

import numpy as np
import os
import librosa
import cPickle
import csv
from scipy import signal
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from scikits.audiolab import wavread
import scipy

from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input
from keras.utils import to_categorical

from util import *
from model import *

def feature_extraction(wav_fd, fe_fd):
    names = [na for na in os.listdir(wav_fd) if na.endswith('.wav')]
    names = sorted(names)
    for na in names:
        print na
        path = wav_fd + '/' + na
        wav, fs, enc = wavread( path )        
        if wav.ndim == 2:
            wav=np.mean(wav, axis=-1)
        ham_win = np.hamming(n_fft)
        [f, t, x] = signal.spectral.spectrogram(x=wav, 
                                                window=ham_win, 
                                                nperseg=n_fft, 
                                                noverlap=0, 
                                                detrend=False, 
                                                return_onesided=True, 
                                                mode='magnitude') 
        x = x.T
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel(sr=fs, 
                                       n_fft=n_fft, 
                                       n_mels=64, 
                                       fmin=0., 
                                       fmax=22100)
        x = np.dot(x, melW.T)
        out_path = fe_fd + '/' + na[0:-4] + '.f'
        cPickle.dump(x, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

def mat_2d_to_3d(X, agg_num, hop):
    # pad to at least one block
    len_X, n_in = X.shape
    if (len_X < agg_num):
        X = np.concatenate((X, np.zeros((agg_num-len_X, n_in))))
        
    # agg 2d to 3d
    len_X = len(X)
    i1 = 0
    X3d = []
    while (i1+agg_num <= len_X):
        X3d.append(X[i1:i1+agg_num])
        i1 += hop
    return np.array(X3d)

def train_data():
    with open( meta_train_csv, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
    X3d_all = []
    y_all = []
    i=0
    for li in lis:
        # load data
        na = li[1]
        path = dev_fd + '/' + na + '.f'
        info_path = label_csv + '/' + na + '.csv'
        with open( info_path, 'rb') as g:
            reader2 = csv.reader(g)
            lis2 = list(reader2)
        tags = lis2[-2][1]

        y = np.zeros( len(labels) )
        for ch in tags:
            y[ lb_to_id[ch] ] = 1
        #i+=1
        #print i
        try:
            X = cPickle.load( open( path, 'rb' ) )
        except Exception as e:
            print 'Error while parsing',path
            continue
        i+=1
        X3d = mat_2d_to_3d( X, agg_num, hop )
        X3d_all.append( X3d )
        y_all += [ y ] * len( X3d )
    
    print 'Files loaded',i
    X3d_all = np.concatenate( X3d_all )
    y_all = np.array( y_all )
    
    return X3d_all, y_all

def test(md):
    y_true=[]
    y_pred=[]
    with open( meta_test_csv, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    names = []
    for li in lis:
        names.append( li[0] )
        na = li[1]
        #audio evaluation name
        fe_path0 = eva_fd + '/' + na + '.f'
        X0 = cPickle.load( open( fe_path0, 'rb' ) )
        X0 = mat_2d_to_3d( X0, agg_num, hop )
        a,b,c=X0.shape
        X0 = X0.reshape(a,1,b,c) #reshape when CNN2D
        info_path = label_csv + '/' + na + '.csv'
        with open( info_path, 'rb') as g:
            reader2 = csv.reader(g)
            lis2 = list(reader2)
        tags = lis2[-2][1]

        y = np.zeros( len(labels) )
        for ch in tags:
            y[ lb_to_id[ch] ] = 1
        y_true.append(y)
        
        p_y_pred = md.predict( X0 )
        p_y_pred = np.mean( p_y_pred, axis=0 ) 
        y_pred.append(p_y_pred)

    y = label_binarize(y, classes=[0,1,2,3,4,5,6,7])
    eps = 1E-6

    n_classes = y.shape[1]
    y_true=np.array(y_true)
    y_pred=np.array(y_pred)
    class_eer=[]

    for k in xrange(n_classes):
        f, t, _ = roc_curve(y_true[:,k], y_pred[:,k])
        Points = [(0,0)]+zip(f,t)
        for i, point in enumerate(Points):
            if point[0]+eps >= 1-point[1]:
                break
        P1 = Points[i-1]; P2 = Points[i]
            
        if abs(P2[0]-P1[0]) < eps:
            ER = P1[0]        
        else:        
            m = (P2[1]-P1[1]) / (P2[0]-P1[0])
            o = P1[1] - m * P1[0]
            ER = (1-o) / (1+m) 
        class_eer.append(ER)

    EER = np.mean(class_eer)
    return EER
#feature_extraction(wav_dev_fd,dev_fd)
#feature_extraction(wav_eva_fd,eva_fd)


train_x,train_y=train_data()

[batch_num, n_time, n_freq] = train_x.shape
train_x=train_x.reshape((batch_num,1,n_time,n_freq))
    
dimx = n_time
dimy = n_freq


model.fit(train_x,train_y,batch_size=batchsize,epochs=2,verbose=1)


eer=test(model)
print "EER %.2f"%eer
