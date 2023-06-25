#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 20:44:32 2023

@author: francisco
"""

import json
import glob
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


import tensorflow as tf

import numpy as np


datasetPath="/home/francisco/Documents/statMLa1/onehotsequencerecords/subsampling/"


data_train_dom1_human=tf.io.TFRecordWriter(datasetPath+'train_dom1_human.record')
data_train_dom1_machine=tf.io.TFRecordWriter(datasetPath+'train_dom1_machine.record')

data_train_dom2_human=tf.io.TFRecordWriter(datasetPath+'train_dom2_human.record')
data_train_dom2_machine=tf.io.TFRecordWriter(datasetPath+'train_dom2_machine.record')

data_val_dom1_human=tf.io.TFRecordWriter(datasetPath+'val_dom1_human.record')
data_val_dom1_machine=tf.io.TFRecordWriter(datasetPath+'val_dom1_machine.record')

data_val_dom2_human=tf.io.TFRecordWriter(datasetPath+'val_dom2_human.record')
data_val_dom2_machine=tf.io.TFRecordWriter(datasetPath+'val_dom2_machine.record')



#data_val=tf.io.TFRecordWriter('/home/francisco/Documents/statMLa1/val.record')
# def token2vec(listTokens): 
#     x=[]
#     for i in range(0,700):
#         vec=np.zeros((5000))
#         if i < len(listTokens):
#             vec[listTokens[i]]=1
#         x.append(vec)
#     return np.array(x).astype("uint8")

def token2vec(listTokens): 
    x=[]
    for i in range(0,700):
        vec=5000*[0]
        if i < len(listTokens):
            vec[listTokens[i]]=1
        x+=vec
    return x

def createTfExample(txt,prompt,label):
    
    
    sample={ 
            'prompt': tf.train.Feature(int64_list=tf.train.Int64List(value= prompt   )),
            'txt': tf.train.Feature(int64_list=tf.train.Int64List(value= txt  )),
             'label':  tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    
    return tf.train.Example(features=tf.train.Features(feature=sample))



np.random.seed(seed=0)

def json2tfrecords(path2files):#use * for pattern names
    #x=[]
    for path in glob.glob(path2files):
        
        f = open(path)
        data = json.load(f)
        f.close()
        
        np.random.shuffle(data)
        
            
        
        for i,sample in tqdm(enumerate(data)):
            seqMatTxt=token2vec(sample["txt"]) #list that has to be reshaped to (700,5000)
            seqMatPrompt=token2vec(sample["prompt"])
            
            
            #x.append(np.concatenate((freqVectorPrompt,freqVectorText)))
            label=1 if "machine" in path else 0
            example=createTfExample(seqMatTxt,seqMatPrompt,label)
            
            if "set1" in path:
                if "human" in path:
                    if i <350:
                        data_val_dom1_human.write(example.SerializeToString())
                    else:
                        if i>=350 and i<3350:
                            data_train_dom1_human.write(example.SerializeToString())
                elif "machine" in path:
                    if i <350:
                        data_val_dom1_machine.write(example.SerializeToString())
                    else:
                        #for _ in range(0,33):
                        data_train_dom1_machine.write(example.SerializeToString())
                        
            elif "set2" in path:
                if "human" in path:
                    if i <40:
                        data_val_dom2_human.write(example.SerializeToString())
                    else:
                        for _ in range(0,6):
                            data_train_dom2_human.write(example.SerializeToString())
                elif "machine" in path:
                    if i <40:
                        data_val_dom2_machine.write(example.SerializeToString())
                    else:
                        data_train_dom2_machine.write(example.SerializeToString())

            #x.append(np.concatenate((freqVectorPrompt,freqVectorText)))
    #return np.array(x)
#human
json2tfrecords('/home/francisco/Documents/statMLa1/dataset/*.json')
#machine
#json2tfrecords('/home/francisco/Documents/statMLa1/comp90051-2023-sem1-proj1/*machine.json')

data_train_dom1_human.close()
data_train_dom1_machine.close()

data_train_dom2_human.close()
data_train_dom2_machine.close()

data_val_dom1_human.close()
data_val_dom1_machine.close()

data_val_dom2_human.close()
data_val_dom2_machine.close()

