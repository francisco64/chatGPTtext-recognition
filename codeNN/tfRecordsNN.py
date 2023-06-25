#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 20:44:32 2023

@author: francisco
"""

import json
import glob
from sklearn.preprocessing import StandardScaler


import tensorflow as tf

import numpy as np


datasetPath="/home/francisco/Documents/statMLa1/simpleNN/finetune/"


data_train_dom1_human=tf.io.TFRecordWriter(datasetPath+'train_dom1_human.record')
data_train_dom1_machine=tf.io.TFRecordWriter(datasetPath+'train_dom1_machine.record')

data_train_dom2_human=tf.io.TFRecordWriter(datasetPath+'train_dom2_human.record')
data_train_dom2_machine=tf.io.TFRecordWriter(datasetPath+'train_dom2_machine.record')

data_val_dom1_human=tf.io.TFRecordWriter(datasetPath+'val_dom1_human.record')
data_val_dom1_machine=tf.io.TFRecordWriter(datasetPath+'val_dom1_machine.record')

data_val_dom2_human=tf.io.TFRecordWriter(datasetPath+'val_dom2_human.record')
data_val_dom2_machine=tf.io.TFRecordWriter(datasetPath+'val_dom2_machine.record')



#data_val=tf.io.TFRecordWriter('/home/francisco/Documents/statMLa1/val.record')

def createTfExample(featureVector,label):

    sample={ #'feature': tf.train.Feature(float_list=tf.train.FloatList(value=featureVector.tolist())),
            #'feature': tf.train.Feature(bytes_list=tf.train.BytesList(value=[featureVector.tobytes()])),
            'feature':  tf.train.Feature(int64_list=tf.train.Int64List(value=np.uint16(featureVector).tolist())),
             'label':  tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    
    return tf.train.Example(features=tf.train.Features(feature=sample))



np.random.seed(seed=0)

def json2tfrecords(path2files):#use * for pattern names
    #x=[]
    for path in glob.glob(path2files):
        print("Processing: ",path)
        f = open(path)
        data = json.load(f)
        f.close()
        
        np.random.shuffle(data)
        
            
        
        for i,sample in enumerate(data):
            freqVectorPrompt=np.zeros((5000))
            freqVectorText=np.zeros((5000))
            for token in sample["prompt"]:
                freqVectorPrompt[token]=1
            for token in sample["txt"]:
                freqVectorText[token]=1
            #x.append(np.concatenate((freqVectorPrompt,freqVectorText)))
            label=1 if "machine" in path else 0
            example=createTfExample(np.concatenate((freqVectorPrompt,freqVectorText)),label)
            
            
            if "set1" in path:
                continue
                if "human" in path:
                    if i <350:
                        data_val_dom1_human.write(example.SerializeToString())
                    else:
                        #if i>=350 and i<3350:
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

