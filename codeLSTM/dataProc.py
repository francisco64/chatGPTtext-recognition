#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 01:26:01 2023

@author: francisco
"""

import json
import numpy as np 
import glob 
import tensorflow as tf

# f = open('/home/francisco/Documents/statMLa1/dataset/set1_machine.json') # data = json.Load(f)
#f.close()

def createTfExample(txt,prompt,label):

    sample={ #'feature': tf.train.Feature(float_list=tf.train.FloatList(value=featureVector.tolist())),
            'prompt': tf.train.Feature(bytes_list=tf.train.BytesList(value=[prompt.tobytes()])),
            'txt': tf.train.Feature(bytes_list=tf.train.BytesList(value=[txt.tobytes()])),
            #'feature':  tf.train.Feature(int64_list=tf.train.Int64List(value=np.uint16(featureVector).tolist())),
             'label':  tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    
    return tf.train.Example(features=tf.train.Features(feature=sample))

def token2vec(listTokens): 
    x=[]
    for i in range(0,700):
        vec=np.zeros((5000))
        if i < len(listTokens):
            vec[listTokens[i]]=1
        x.append(vec)
    return np.array(x).astype("uint8")

data=tf.data.TFRecordDataset(glob.glob('/home/francisco/Documents/statMLa1/dataset/set2_human.json'))#check if it is better use a falanced version of the dataset that has participation of both domains or a metric that outweights both domains


for path in glob.glob('/home/francisco/Documents/statMLa1/dataset/*.json'):
    f = open(path)
    data= json.load(f)
    f.close()
    label=1 if "machine" in path else 0
    for sample in data:
        seqMatTxt=token2vec (sample["txt"])
        seqMatPrompt=token2vec (sample["prompt"])
        example=createTfExample(seqMatTxt,seqMatPrompt,label)
            
    break
    