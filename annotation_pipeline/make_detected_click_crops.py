# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 05:34:25 2020

@author: jkakl
"""



'''
Use global times to generate new 1-sec audio file crops where each detected click time 
is centered in the middle of the crop.

'''



import numpy as np
import pickle
import math
import librosa
import librosa.display
from scipy.io import wavfile
import io, os
import time
import argparse
import shutil





def rootname_to_book_num(audio_rootname, books):
    for book_num in range(len(books)):
        book, filename = books[book_num]
        if audio_rootname == filename:
            return book_num


def get_directory(directory, rootname):
    years = [2014, 2015, 2016, 2017, 2018]
    file_directory = None
    for yr in years:
        file_directory = directory + str(yr) + '/'
        if rootname + '.wav' in os.listdir(file_directory):
            file_directory = file_directory + rootname + '.wav'
            break
    return file_directory
    
    
def get_year(file_directory):
    years = [2014, 2015, 2016, 2017, 2018]
    for yr in years:
        if str(yr) in file_directory:
            return yr


def crop_audio_clicks(file_to_crop = None, click_time_preds = None, save_dir = None):
    '''
    Parameters
    ----------
    file_to_crop : string
        DESCRIPTION. audio file (rootname) predictions refer to.
    click_time_preds : list
        DESCRIPTION. list of doubles (click time detections).

    Returns
    -------
    List
    Directory annotations of these crops to feed into click separator.
    
    Effect: Generates new 1-sec audio file crops where each detected click time 
    is centered in the middle of the crop. Saves them as '.wav' files.
    Naming: i-th detected click -> 'audio_rootname_[6 digits of i].wav' 

    '''
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)    
    
    detected_clicks_annot = []
    
    dataset_directory = '/data/scratch/ioannis/dataset/'
    file_directory = get_directory(dataset_directory, file_to_crop)
    yr = get_year(file_directory)
    
    new_crops_dir = save_dir + str(yr) + '/' + file_to_crop + '/' 
    if not os.path.exists(new_crops_dir):
        os.makedirs(new_crops_dir)
    
    print(file_directory)
    print('Making detected click crops for file ' + file_to_crop + ', year ' + str(yr))
        
    data, rate = librosa.load(file_directory, mono=False) ## rate = 22050
    print('finished loading audio file..')
    
    #### remove '.wav' ####
    # filename = file_to_crop[:-4]       
    
    ## they remain sorted, no duplicates
    ## assuming rate = 22050
    click_preds_positives = [pred * rate for pred in click_time_preds]
    
    for i in range(len(click_preds_positives)):        
        if i % 500 == 0:
            print(i)
        
        pred_pos = click_preds_positives[i]
        #################################################################################
        new_crop = data[:, int(pred_pos - 0.5*rate) : int(pred_pos + 0.5*rate)]
        #################################################################################
        
        i_str = (6-len(str(i)))*'0' + str(i)
        new_file = os.path.join(new_crops_dir, file_to_crop + '_' + i_str + '.wav')
        
        librosa.output.write_wav(new_file, new_crop, rate)
        
        if pred_pos / rate != click_time_preds[i]:
            print('issue: ', pred_pos, click_time_preds[i])
            
        detected_clicks_annot.append((new_file, click_time_preds[i]))
    
    
    # pickle.dump(detected_clicks_annot, open('/data/vision/torralba/scratch/ioannis/clustering/click_regress_all_admin/detected_clicks_annot_data_list_' + audio_rootname + '.p', 'wb'))
    
    detected_clicks_annot = np.array(detected_clicks_annot, dtype=object)    
    return detected_clicks_annot
        
        





