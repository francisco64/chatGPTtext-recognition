#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 20:44:32 2023

@author: francisco
"""

import json
import glob
from sklearn.preprocessing import StandardScaler
#from multiprocessing import Pool

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
def token2vec(listTokens): 
    x=[]
    for i in range(0,700):
        vec=np.zeros((5000))
        if i < len(listTokens):
            vec[listTokens[i]]=1
        x.append(vec)
    return np.array(x).astype("uint8")



def createTfExample(txt,prompt,label):
    
    
    sample={ 
            'prompt': tf.train.Feature(int64_list=tf.train.Int64List(value= prompt.flatten().tolist()   )),
            'txt': tf.train.Feature(int64_list=tf.train.Int64List(value= txt.flatten().tolist()   )),
             'label':  tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    
    return tf.train.Example(features=tf.train.Features(feature=sample))



np.random.seed(seed=0)

def examplePromtText(sample,label):
    seqMatTxt=token2vec (sample["txt"])
    seqMatPrompt=token2vec (sample["prompt"])
    return createTfExample(seqMatTxt,seqMatPrompt,label)

def writeTfrecords(path,i, sample):
        
        
        #x.append(np.concatenate((freqVectorPrompt,freqVectorText)))
        label=1 if "machine" in path else 0
       
        
        
        if "set1" in path:
            if "human" in path:
                if i <350:
                    example=examplePromtText(sample,label)
                    data_val_dom1_human.write(example.SerializeToString())
                else:
                    if i>=350 and i<3350:
                        example=examplePromtText(sample,label)
                        data_train_dom1_human.write(example.SerializeToString())
                    else: 
                        return
            elif "machine" in path:
                example=examplePromtText(sample,label)
                if i <350:
                    
                    data_val_dom1_machine.write(example.SerializeToString())
                else:
                    #for _ in range(0,33):
                    
                    data_train_dom1_machine.write(example.SerializeToString())
                    
        elif "set2" in path:
            example=examplePromtText(sample,label)
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


def json2tfrecords(path2files):#use * for pattern names
    #x=[]
    
    
    
    for path in glob.glob(path2files):
        print("processing: ",path)
        #pool=Pool()#processes=8)
        f = open(path)
        data = json.load(f)
        f.close()
        
        np.random.shuffle(data)
        
        for i,sample in enumerate(data):
            
            #pool.apply_async(writeTfrecords, (path,i, sample,))
            writeTfrecords(path,i, sample)
            
        #pool.close()
        #pool.join()
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

