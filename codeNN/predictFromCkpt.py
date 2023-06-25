#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 03:16:22 2023

@author: francisco
"""




import glob,json
import numpy as np
import tensorflow as tf


def json2vector(path2files):#use * for pattern names
    x=[]
    for path in glob.glob(path2files):
        f = open(path)
        data = json.load(f)
        f.close()
        
        for sample in data:
            freqVectorPrompt=np.zeros((5000))
            freqVectorText=np.zeros((5000))
            for token in sample["prompt"]:
                freqVectorPrompt[token]=1
            for token in sample["txt"]:
                freqVectorText[token]=1
            x.append(np.concatenate((freqVectorPrompt,freqVectorText)))
    return np.array(x)

x=json2vector('/home/francisco/Documents/statMLa1/test.json')



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

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

model.load_weights( '/home/francisco/Documents/statMLa1/simpleNN/finetune/evaluatedD1+D2/checkpoint/model.3-best.h5')



import pandas as pd




feature_format={
    'feature':  tf.io.FixedLenFeature([10000],tf.int64),#tf.io.FixedLenFeature([],tf.string),
    'label':  tf.io.FixedLenFeature([1],tf.int64)
}
def get_dataFromBinary(proto):
    f=tf.io.parse_single_example(proto,feature_format)
    #feature=tf.io.decode_raw(f['feature'],np.float64)#tf.float64)
    #feature=tf.reshape(feature,[1,10000])
    feature=tf.reshape(f['feature'],[10000])
    label=f['label']
    return (feature,label)

def proc_dataset(dataset):
    #dataset=datasetWoBatch.map(normalizeDataset)
    dataset=dataset.map(get_dataFromBinary)
    dataset = dataset.shuffle(32)
    dataset=dataset.batch(100)
    return dataset


print("-----------------Model Evaluation---------------")
data_val=proc_dataset(tf.data.TFRecordDataset(glob.glob('/home/francisco/Documents/statMLa1/simpleNN/original/val*.record')))
print("whole validation")
model.evaluate(data_val)

print("Domain 1:")
model.evaluate(proc_dataset(tf.data.TFRecordDataset(glob.glob('/home/francisco/Documents/statMLa1/simpleNN/original/val_dom1*.record'))))
print("Domain 2:")
model.evaluate(proc_dataset(tf.data.TFRecordDataset(glob.glob('/home/francisco/Documents/statMLa1/simpleNN/original/val_dom2*.record'))))


y=[]
for f,l in data_val:
    for label in l:
        y.append(label)
y=np.array(y)
print(y)
pred_th=[ np.sum(np.uint8(model.predict(data_val)>th) == y) for th in np.arange(0,1,0.1)]

print(pred_th)

optTh=np.arange(0,1,0.1)[np.argmax(pred_th)]#check
                    
print("threshold optimo ",optTh)
predictions=np.uint8(model.predict(x)<optTh)

pd.DataFrame(data=[(i,p[0]) for i,p in enumerate(predictions)],columns=("Id","Predicted")).to_csv('/home/francisco/Documents/statMLa1/predictions.csv')
