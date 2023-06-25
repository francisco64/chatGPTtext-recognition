#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:54:45 2023

@author: francisco
"""



import tensorflow as tf
import numpy as np
import glob



feature_format={
    'prompt':  tf.io.FixedLenFeature([700*5000],tf.int64),
    'txt':  tf.io.FixedLenFeature([700*5000],tf.int64),#tf.io.FixedLenFeature([],tf.string),
    'label':  tf.io.FixedLenFeature([1],tf.int64)
}




def get_dataFromBinary(proto):
    f=tf.io.parse_single_example(proto,feature_format)
    #feature=tf.io.decode_raw(f['feature'],np.float64)#tf.float64)
    #feature=tf.reshape(feature,[1,10000])
    
    prompt=tf.reshape(f['prompt'],[700,5000])
    txt=tf.reshape(f['txt'],[700,5000])
    label=f['label']
    return ({"prompt":prompt,"txt":txt},label)#((prompt,txt),label)


def proc_dataset(dataset):
    #dataset=datasetWoBatch.map(normalizeDataset)
    dataset=dataset.map(get_dataFromBinary)
    dataset = dataset.shuffle(100)
    dataset=dataset.batch(10)
    return dataset

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,concatenate
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import LSTM

def classifier():
         
    inputPrompt = tf.keras.layers.Input(shape=(700,5000),name="inputPrompt")
    inputTxt = tf.keras.layers.Input(shape=(700,5000),name="inputTxt")
    
    embeddingPrompt=Dropout(0.5)(LSTM(32, activation='relu')(inputPrompt))
    promptLayer = tf.keras.Model(inputs=inputPrompt, outputs=embeddingPrompt,name="promptEmbedding")
    
    embeddingTxt=Dropout(0.5)(LSTM(32, activation='relu')(inputTxt) )
    txtLayer = tf.keras.Model(inputs=inputTxt, outputs=embeddingTxt,name="txtEmbedding")
    
    combinedEmbedding = concatenate([promptLayer.output, txtLayer.output])
    
    outputEmbedding = Dense(20, activation="relu")(combinedEmbedding)
    
    out = Dense(1,activation='sigmoid')(outputEmbedding)
    
    model = tf.keras.Model(
    inputs={"prompt":inputPrompt,"txt":inputTxt},
    outputs=out,
    )
    

    
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.AUC(name='auc'),tf.keras.metrics.AUC(name='prc', curve='PR')])
    return model

data_train=tf.data.TFRecordDataset(glob.glob('/home/francisco/Documents/statMLa1/onehotsequencerecords/subsampling/train_*.record') )



data_val=tf.data.TFRecordDataset(glob.glob('/home/francisco/Documents/statMLa1/onehotsequencerecords/subsampling/val_*.record'))


dataset_train=proc_dataset(data_train)
dataset_val=proc_dataset(data_val)





# def classifier():
#   model=Sequential()
#   model.add(Dense(200,input_dim=(10000),activation='relu'))
#   model.add(Dense(180,activation='relu'))
#   model.add(Dropout(0.5))
#   model.add(Dense(150,activation='relu'))
#   model.add(Dense(10,activation='relu'))
#   model.add(Dense(1,activation='sigmoid'))
#   model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.AUC(name='auc'),tf.keras.metrics.AUC(name='prc', curve='PR')])
#   return model
     



checkpoint=ModelCheckpoint('/home/francisco/Documents/statMLa1/checkpoint/model.{epoch:d}.h5',save_best_only=False, save_freq='epoch')
tensorboard_callback=TensorBoard('/home/francisco/Documents/statMLa1/logs',histogram_freq=1)


# # total=122e3 + 3.5e3 + 100 + 400 
# # neg=122e3 + 100
# # pos=3.5e3 + 400

# # weight_for_0 = (1 / neg) * (total / 2.0)
# # weight_for_1 = (1 / pos) * (total / 2.0)

# # class_weight = {0: weight_for_0, 1: weight_for_1}

model=classifier()

# # for sample in dataset_train:
# #     print(sample)
# #     break
model.fit(dataset_train,epochs=10, batch_size=1000,validation_data=dataset_val,callbacks=[tensorboard_callback,checkpoint])#,class_weight=class_weight)





