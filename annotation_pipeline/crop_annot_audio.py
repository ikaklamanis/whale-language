# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:31:11 2020

@author: jkakl
"""

'''' 
Code to split/crop audio files for which there are annotations.


'''



import numpy as np
import pickle
import librosa
import librosa.display
import io, os
import time
import shutil




def get_directory(directory, filename):
    years = [2014, 2015, 2016, 2017, 2018]
    file_directory = None
    for yr in years:
        file_directory = directory + str(yr) + '/'
        if filename + '.wav' in os.listdir(file_directory):
            file_directory = file_directory + filename + '.wav'
            break
    return file_directory
    
    
def get_year(file_directory):
    years = [2014, 2015, 2016, 2017, 2018]
    for yr in years:
        if str(yr) in file_directory:
            return yr


def crop_audio(file_to_crop, dataset_directory, cropped_audio_save_dir):
    
    file_directory = get_directory(dataset_directory, file_to_crop)
    yr = get_year(file_directory)
    
    print(file_directory)
    print('Processing file ' + file_to_crop + ', year ' + str(yr))        
    data, rate = librosa.load(file_directory, mono=False) ## rate = 22050    
    print('finished loading audio file..')
    
    #### make naming correct (remove .wav) ####
    # filename = file_to_crop[:-4]
    filename = file_to_crop
    
    new_file_dir_1 = os.path.join(cropped_audio_save_dir, str(yr), filename)
    
    if not os.path.exists(new_file_dir_1):
        os.makedirs(new_file_dir_1)
        
    for j in range(0, int(data.shape[1] / rate)):
        if j % 500 == 0:
            print(j)
            
        new_data = data[:, j*rate : (j+1)*rate]
        
        j_str = (6-len(str(j)))*'0' + str(j)
        new_file_1 = os.path.join(new_file_dir_1, filename + '_' + j_str + '.wav')        
        librosa.output.write_wav(new_file_1, new_data, rate)



def create_annot_audio_dirs_augmented(file_to_crop, cropped_audio_save_dir, sort=True, augment=True):
    
    dataset_directory = '/data/scratch/ioannis/dataset/'    
    # root_dir = get_directory(cropped_audio_save_dir, file_to_crop[:-4])
    old_file_directory = get_directory(dataset_directory, file_to_crop)
    yr = get_year(old_file_directory)
    root_dir = os.path.join(cropped_audio_save_dir, str(yr), file_to_crop)
        
    file_paths = [(root_dir + '/' + file, [] ) for file in os.listdir(root_dir)]
    ## sort file paths based on ext number ##
    if sort:
        file_paths.sort(key = lambda file_path: int(file_path[0].split('/')[-1][10:16]))
    ## augment with prev and next audio file directory if they exist, else None ##
    if augment:
        new_file_paths = []
        for i in range(len(file_paths)):
            curr_file = file_paths[i][0]
            prev_file = None if i == 0 else file_paths[i-1][0]
            next_file = None if i == len(file_paths) - 1 else file_paths[i+1][0]
            new_file_paths.append(((prev_file, curr_file, next_file), []))
        file_paths = new_file_paths
    ## end of augmentation ## 
        
    return file_paths





