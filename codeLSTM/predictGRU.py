#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 03:16:22 2023

@author: francisco
"""




import glob,json
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,concatenate,Embedding
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import GRU

num_words=5000



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


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
    x=Dense(20,activation='relu')(x)
    pred = Dense(1,activation='sigmoid',name="output")(x)
    
    model = tf.keras.Model(
        inputs=[prompt_input, txt_input],
        outputs=pred,
    )

    model.compile(loss=['binary_crossentropy'], optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.AUC(name='auc'),tf.keras.metrics.AUC(name='prc', curve='PR')])
  

    return model

model=classifier()

model.load_weights('/home/francisco/Documents/statMLa1/checkpoint/model.1655.h5')



import pandas as pd


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

pad=lambda a: np.array(a[0:700] if len(a) > 700 else a + [0] * (700-len(a)))

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

valPrompt,valtxt,y_val=toArrays(val_d1+val_d2)

valPrompt_d1,valtxt_d1,y_val_d1=toArrays(val_d1)

valPrompt_d2,valtxt_d2,y_val_d2=toArrays(val_d2)




print("-----------------Model Evaluation---------------")
print("whole validation")
model.evaluate([valPrompt,valtxt],y_val)

print("Domain 1:")
model.evaluate([valPrompt_d1,valtxt_d1],y_val_d1)
print("Domain 2:")
model.evaluate([valPrompt_d2,valtxt_d2],y_val_d2)



y_val=np.array(y_val)
print(model.predict([valPrompt,valtxt]))
import matplotlib.pyplot as plt

plt.hist(model.predict([valPrompt,valtxt]), bins=100)
plt.show()


pred_th=[]
for th in np.arange(0,1,0.1):
    prediction=np.uint8(model.predict([valPrompt,valtxt])>th)
    val=0
    for i in range(0,len(y_val)):
        
        if prediction[i]==y_val[i]:
            val+=1
    pred_th.append(val/len(y_val))

print(pred_th)

optTh=np.arange(0,1,0.1)[np.argmax(pred_th)]#check
                    
print("threshold optimo ",optTh)

f = open('/home/francisco/Documents/statMLa1/test.json')
data= json.load(f)
f.close()
testPrompt=[]
testtxt=[]
for d in data:
    testPrompt.append(pad(list(d["prompt"])) )
    testtxt.append(pad(list(d["txt"])))

testPrompt=np.array(testPrompt)
testtxt=np.array(testtxt)


predictions=np.uint8(model.predict([testPrompt,testtxt])<optTh)

pd.DataFrame(data=[(i,p[0]) for i,p in enumerate(predictions)],columns=("Id","Predicted")).to_csv('/home/francisco/Documents/statMLa1/predictions.csv')
