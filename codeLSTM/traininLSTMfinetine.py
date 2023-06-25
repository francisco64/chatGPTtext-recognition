#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:54:45 2023

@author: francisco
"""



import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import numpy as np
import glob

num_words = 5000

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,concatenate,Embedding
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import LSTM,GRU


def classifier():
    prompt_input = tf.keras.Input(
        shape=(None,), name="prompt"
    ) 
    
    txt_input = tf.keras.Input(
        shape=(None,), name="txt"
    ) 
    
    # Embed each word in the prompt into a 64-dimensional vector
    prompt_features = Embedding(num_words, 64)(prompt_input)
    # Embed each word in the text into a 64-dimensional vector
    txt_features = Embedding(num_words, 64)(txt_input)
    
    prompt_features =Dropout(0.2)( GRU(32)(prompt_features))
    
    
    txt_features =Dropout(0.2)( GRU(32)(txt_features))
    
    x=concatenate([prompt_features, txt_features])
    
    pred = Dense(1,activation='sigmoid',name="output")(x)
    
    model = tf.keras.Model(
        inputs=[prompt_input, txt_input],
        outputs=pred,
    )

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.AUC(name='auc'),tf.keras.metrics.AUC(name='prc', curve='PR')])
  

    return model



#tf.keras.utils.plot_model(model, "/home/francisco/Documents/statMLa1/codeLSTM/multi_input_and_output_model.png", show_shapes=True)

  




checkpoint=ModelCheckpoint('/home/francisco/Documents/statMLa1/checkpoint/model.{epoch:d}.h5',save_best_only=False, save_freq='epoch')
tensorboard_callback=TensorBoard('/home/francisco/Documents/statMLa1/logs',histogram_freq=1)


# total=122e3 + 3.5e3 + 100 + 400 
# neg=122e3 + 100
# pos=3.5e3 + 400

# weight_for_0 = (1 / neg) * (total / 2.0)
# weight_for_1 = (1 / pos) * (total / 2.0)

# class_weight = {0: weight_for_0, 1: weight_for_1}

#model=classifier()

# # for sample in dataset_train:
# #     print(sample)
# #     break
import json

import glob

np.random.seed(seed=0)


val_d1=[]
val_d2=[]
train_d1=[]
train_d2=[]


for path in glob.glob('/home/francisco/Documents/statMLa1/dataset/*.json'):
    f = open(path)
    data= json.load(f)
    f.close()
    
    np.random.shuffle(data)
        
    for i, sample in enumerate(data):
        txt=sample["txt"]
        prompt=sample["prompt"]


        label=1 if "machine" in path else 0
    
    
        if "set1" in path:
            if "human" in path:
                if i <350:
                    val_d1.append(({"prompt":np.array(prompt),"txt":np.array(txt)},label))
                else:
                    if i>=350 and i<3350:
                        train_d1.append(({"prompt":np.array(prompt),"txt":np.array(txt)},label))
                
                
            elif "machine" in path:
                    if i <350:
                        
                        val_d1.append(({"prompt":np.array(prompt),"txt":np.array(txt)},label))
                        
                    else:
                        #for _ in range(0,33):
                        train_d1.append(({"prompt":np.array(prompt),"txt":np.array(txt)},label))

        elif "set2" in path:
            if "human" in path:
                if i <40:
                    val_d2.append(({"prompt":np.array(prompt),"txt":np.array(txt)},label))
                else:
                        #if i>=350 and i<3350:
                    for _ in range(0,6):        
                        train_d2.append(({"prompt":np.array(prompt),"txt":np.array(txt)},label))
                
                
            elif "machine" in path:
                    if i <40:
                        
                        val_d2.append(({"prompt":np.array(prompt),"txt":np.array(txt)},label))
                        
                    else:
                        #for _ in range(0,33):
                        train_d2.append(({"prompt":np.array(prompt),"txt":np.array(txt)},label))

pad=lambda a: np.array(a[0:700] if len(a) > 700 else a + [1] * (700-len(a)))

def toArrays(data):
    trainPrompt=[]
    y_train=[]
    traintxt=[]
    np.random.shuffle(data)
    for d,l in data:
        
        trainPrompt.append(pad(list(d["prompt"])) )
        traintxt.append(pad(list(d["txt"])))
        y_train.append(l)
    
    trainPrompt=np.array(trainPrompt)
    traintxt=np.array(traintxt)
    y_train=np.array(y_train)   
    return trainPrompt,traintxt,y_train

trainPrompt,traintxt,y_train=toArrays(train_d2)

valPrompt,valtxt,y_val=toArrays(val_d2)

model=classifier()


model.load_weights('/home/francisco/Documents/statMLa1/onehotsequencerecords/subsampling/GRU/checkpoint/model.159.h5')

model.summary()
for layer in model.layers:
    
    if layer.name in ['output']:
        layer.trainable = True
    else:
        layer.trainable = False
model.summary()
model.fit([trainPrompt,traintxt],y_train,epochs=100, batch_size=5,validation_data=([valPrompt,valtxt],y_val),callbacks=[tensorboard_callback,checkpoint])#,class_weight=class_weight)

