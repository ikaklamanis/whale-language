# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:01:43 2020

@author: jkakl
"""


import pickle
import numpy as np
import argparse
import os


def get_year(file_path):
    yr = int(file_path.split('/')[-3])
    return yr
    
def get_rootfile(file_path):
    root = str(file_path.split('/')[-2])
    return root

def get_ext(file_path):
    file_ext = str(file_path.split('/')[-1][:-4])
    return file_ext


def relocate_predictions(audio_rootname, audio_annot_directory, det_save_dir, reloc_dir):
    
    if not os.path.exists(reloc_dir):
        os.makedirs(reloc_dir)
    
    audio_annot_dirs = pickle.load(open(audio_annot_directory,"rb"))
    data_split = audio_annot_dirs ## custom: working with no groups
    print('data_split: ', data_split.shape)
    batch_size = data_split.shape[0]
    print('batch size: ', batch_size)
    
    batch_files = [file for file in os.listdir(det_save_dir)]
    ## extract '0' from 'batch_0.p'
    batch_numbers = sorted([int(file.split('.')[0].split('_')[1]) for file in batch_files])
    first_batch, last_batch = batch_numbers[0], batch_numbers[-1]
    
    print('first batch: ', first_batch)
    print('last batch: ', last_batch)
    
    j = first_batch * batch_size ## custom (correct)
    
    for i_batch in range(first_batch, last_batch + 1): ## attention: need to parse in order ## issue: not sorted
        print('i_batch: ', i_batch)
        data = pickle.load(open(det_save_dir + 'batch_' + str(i_batch) + '.p', 'rb'))
        batch_size = data.shape[0]
        
        for k in range(batch_size):
            # print(j)            
            file_path, labels = data_split[j, 0], data_split[j, 1]
            file_path_1 = file_path[1] ### VERY CUSTOM, need it because I have triple of audio_dirs
            year = get_year(file_path_1)
            root_file = get_rootfile(file_path_1)
            file_ext = get_ext(file_path_1)
            # print(file_ext, file_ext[-6:], j)            
                        
            save_dir = reloc_dir + str(year) + '/' + str(root_file) + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            preds = data[k : k+1, :]
            #### SOS ####
            info = (file_path, preds, labels) ## file_path = audio_dir
            #### SOS ####
            pickle.dump(info, open(save_dir + file_ext + '.p', 'wb'))
            
            j += 1    



    