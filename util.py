# -*- coding: utf-8 -*-
"""
** deeplean-ai.com **
** dl-lab **
created by :: akshitac8
"""

#Tell your wavs folder
wav_dev_fd = "D:/workspace/aditya_akshita/datax/chime/audios/development"
wav_eva_fd = "D:/workspace/aditya_akshita/datax/chime/audios/evaluation"
dev_fd = "D:/workspace/aditya_akshita/datax/chime/features/development"
eva_fd = "D:/workspace/aditya_akshita/datax/chime/features/evaluation"
meta_train_csv = 'D:/workspace/aditya_akshita/datax/chime/texts/meta_csvs/development_chunks_refined.csv'
meta_test_csv  = 'D:/workspace/aditya_akshita/datax/chime/texts/meta_csvs/evaluation_chunks_refined.csv'
label_csv = 'D:/workspace/aditya_akshita/datax/chime/texts/label_csvs'
labels = [ 'c', 'm', 'f', 'v', 'p', 'b', 'o', 'S' ]
            
lb_to_id = {lb:id for id, lb in enumerate(labels)}
id_to_lb = {id:lb for id, lb in enumerate(labels)}

fsx = 44100.
n_fft = 1024.
agg_num=10
hop=10
num_classes=len(labels)

input_neurons=200
act1='relu'
act2='relu'
act3='softmax'
epochs=100
batchsize=100
nb_filter = 100
filter_length =3
pool_size=(2,2)
