#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:54:45 2023

@author: francisco
"""



import tensorflow as tf
import numpy as np
import glob




def get_dataFromBinary(proto):
    f=tf.io.parse_single_example(proto,feature_format)
    #feature=tf.io.decode_raw(f['feature'],np.float64)#tf.float64)
    #feature=tf.reshape(feature,[1,10000])
    feature=tf.reshape(f['feature'],[10000])
    label=f['label']
    return (feature,label)

#datasetWoBatch=data.map(get_dataFromBinary)

# maxVec=np.array(10000*[1e-10])
# minVec=np.zeros((10000))

# for f,l in datasetWoBatch:
#     f=f.numpy()
    
#     for i, col in enumerate(f):
#         if col>maxVec[i]:
#             maxVec[i]=col
#         if col<minVec[i]:
#             minVec[i]=col
    
# def normalizeDataset(dataset):
#     feature,label=dataset
#     feature=(feature-minVec)/(maxVec-minVec)
#     return (feature,label)
#meanVect=np.mean(np.float32(list(datasetWoBatch.as_numpy_iterator())[0]))

#stdVect=np.std(list(datasetWoBatch.as_numpy_iterator())[1])

def proc_dataset(dataset):
    #dataset=datasetWoBatch.map(normalizeDataset)
    dataset=dataset.map(get_dataFromBinary)
    dataset = dataset.shuffle(32)
    dataset=dataset.batch(100)
    return dataset


data_train=tf.data.TFRecordDataset(glob.glob('/home/francisco/Documents/statMLa1/simpleNN/finetune/train*.record'))

data_val=tf.data.TFRecordDataset(glob.glob('/home/francisco/Documents/statMLa1/simpleNN/original/val_*.record'))#check if it is better use a falanced version of the dataset that has participation of both domains or a metric that outweights both domains
#data_val=tf.data.TFRecordDataset(glob.glob('/home/francisco/Documents/statMLa1/simpleNN/undersampling/val_*.record'))#check if it is better use a falanced version of the dataset that has participation of both domains or a metric that outweights both domains

feature_format={
    'feature':  tf.io.FixedLenFeature([10000],tf.int64),#tf.io.FixedLenFeature([],tf.string),
    'label':  tf.io.FixedLenFeature([1],tf.int64)
}

dataset_train=proc_dataset(data_train)
dataset_val=proc_dataset(data_val)

# for sample in dataset_train:
#     print(sample)
#     break


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

def classifier():
  model=Sequential()
  model.add(Dense(200,input_dim=(10000),activation='relu'))
  model.add(Dense(180,activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(150,activation='relu'))
  model.add(Dense(10,activation='relu'))
  model.add(Dense(1,activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.AUC(name='auc'),tf.keras.metrics.AUC(name='prc', curve='PR')])
  return model
     
model=classifier()
model.load_weights('/home/francisco/Documents/statMLa1/simpleNN/original/checkpoint/model.20-best.h5')


checkpoint=ModelCheckpoint('/home/francisco/Documents/statMLa1/checkpoint/model.{epoch:d}.h5',save_best_only=False, save_freq='epoch')
tensorboard_callback=TensorBoard('/home/francisco/Documents/statMLa1/logs',histogram_freq=1)
# model.trainable=False
for layer in model.layers:
    if layer.name in ['dense_2', 'dense_3','dense_4']:
        layer.trainable = True
    else:
        layer.trainable = False

# total=400+100
# neg=100 
# pos=400

# weight_for_0 = (1 / neg) * (total / 2.0)
# weight_for_1 = (1 / pos) * (total / 2.0)

# class_weight = {0: weight_for_0, 1: weight_for_1}


model.summary()

model.fit(dataset_train,validation_data=dataset_val,epochs=100, batch_size=5,callbacks=[tensorboard_callback,checkpoint])#,class_weight=class_weight)



#model.trainable=False
