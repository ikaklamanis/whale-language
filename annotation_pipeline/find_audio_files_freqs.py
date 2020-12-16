# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 01:21:50 2020

@author: jkakl
"""

import pickle
import numpy as np

def find_files_frequency_in_train_data():
    data_directory = '/data/scratch/ioannis/new_annotation.p'
    total_data = pickle.load(open(data_directory,"rb"))
    
    files_freqs = {}    
    train_data = total_data[: int(0.8*total_data.shape[0])]
    
    for i in range(train_data.shape[0]):
        audio_dir = train_data[i,0]
        rootname = audio_dir.split('/')[-1].split('_')[0]
        if rootname in files_freqs:
            files_freqs[rootname] += 1
        else:
            files_freqs[rootname] = 1
    
    return files_freqs


files_freqs = find_files_frequency_in_train_data()

print('files frequencies: ')
print(files_freqs)
    
    
    