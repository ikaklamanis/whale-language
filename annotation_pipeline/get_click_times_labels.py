# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 19:06:04 2020

@author: jkakl
"""

import numpy as np
import os
from numpy import genfromtxt 
import pickle
import pandas as pd
from scipy import ndimage
import csv





# Function: parse de coda and get basic info
def parseCoda(book, i, time_origin):
    if i==-1:
        return [0,0,0,0,0,0]
    coda = book[i,:]
    whale_number = coda[0].astype(int)
    t_init = coda[1]-time_origin  
    click_times = coda[2:42].astype(float).tolist()
    num_clicks = np.count_nonzero(click_times)+1 # sum 1 because the first click is always zero.
    click_times = click_times[0:num_clicks]
    average_power = coda[84]
    # 43rd column to 83rd column: Powers of the clicks
    click_power = coda[43:83].astype(float).tolist()
    click_power = np.array(click_power[0:num_clicks])
    return [whale_number, t_init, num_clicks, click_times, average_power, click_power]


def rootname_to_book_num(audio_rootname, books):
    for book_num in range(len(books)):
        book, filename = books[book_num]
        if audio_rootname == filename:
            return book_num
        

def save_ground_truth_click_times(audio_rootname):
    
    main_dir = '/data/vision/torralba/scratch/ioannis/clustering/'
    
    annotations_dir = '/data/scratch/ioannis/click_seperator/list_annotations.p'
    books = pickle.load(open(annotations_dir, 'rb'))
    
    book_num = rootname_to_book_num(audio_rootname, books)    
    book, filename = books[book_num]       
            
    print('Processing book ', str(book_num), ' filename ', filename)
    print('---------------------------------')
    num_stories = book.shape[0]
    print('num stories: ', num_stories)
    total_clicks = 0
    file_clicks = []
    all_times = []
    
    
    rootname = filename.split('/')[-1][:6] ## get rootname of file, check for match
    ext = int(filename.split('/')[-1][6:9])            
    
    # time_origin = time_dict[rootname]['time']
    time_origin = 0
    
    for i in range(num_stories):
        [whale_number, t_init, num_clicks, click_times, average_power, click_power] = parseCoda(book, i, time_origin)
        total_clicks += num_clicks
        
        for j in range(num_clicks):
            click_time = t_init + click_times[j]            
            file_clicks.append((whale_number, click_time))
            all_times.append(click_time)
    
    print('file clicks length: ', len(file_clicks))    
    
    file_clicks.sort(key = lambda pair : pair[1]) ## sort based on click time
    custom_labels_save_dir = main_dir + 'click_regress_all_admin/' + filename + '_labels.p'
    pickle.dump(file_clicks, open(custom_labels_save_dir, 'wb'))    
        



def save_global_and_local_labels(audio_rootname):
    
    ## window in seconds: 2000 / 22050
    ## only considers next (not prev)
    ## returns singleton list or length-2 list
    def time_to_indices(click_time = 0, window = 0.091):
        indices = [int(click_time)]
        if click_time - int(click_time) < window and int(click_time) >= 1:
            indices.append(int(click_time) - 1)
        return indices
    
    annotations_dir = '/data/scratch/ioannis/click_seperator/list_annotations.p'
    books = pickle.load(open(annotations_dir, 'rb'))    
    key = rootname_to_book_num(audio_rootname, books)
    
    print('-------------', audio_rootname, ', key: ', str(key), '-------------')
    
    main_dir = '/data/vision/torralba/scratch/ioannis/clustering/'
    
    ## audio file directories of the form (prev/None, current, next/None) for entire 2-hr file (triples)
    test_pick = '/data/scratch/ioannis/' + 'click_regress_all_admin/' + 'annot_audio_' + audio_rootname + '.p' ## also triples
    test_pick = main_dir + 'click_regress_all_admin/' + 'annot_audio_' + audio_rootname + '.p' ## also triples
    audio_annot = pickle.load(open(test_pick,"rb"))
    
    labels_dir = '/data/scratch/ioannis/' + 'click_regress_all_admin/' + audio_rootname + '_labels.p'
    labels_dir = main_dir + 'click_regress_all_admin/' + audio_rootname + '_labels.p'
    file_clicks = pickle.load(open(labels_dir, 'rb'))    
    
    #############################################################
    data_ordered_dir = '/data/vision/torralba/scratch/ioannis/clustering/click_separator_training/file_ordered_correct_times.p'
    data_ordered = pickle.load(open(data_ordered_dir, 'rb'))
    ## VERY CUSTOM WAY TO GET duration of sw061b001, works for sw061b001, sw061b002
    # prev_duration = file_clicks[0][1] - float(data_ordered[str(key)][0][2])    
    
    file = '/data/vision/torralba/scratch/ioannis/clustering/click_regress_all_admin/audio_files_durations.p'
    durations = pickle.load(open(file, 'rb'))
    prev_duration = 0
    try:
        rootname = audio_rootname[:6] ## get rootname of file, check for match
        ext = int(audio_rootname[6:9])
        if ext >= 2:
            for prev_ext in range(1, ext):
                prev_ext_str = (3-len(str(prev_ext)))*'0' + str(prev_ext)
                prev_file = rootname + prev_ext_str
                prev_duration += durations[prev_file]['duration']
    except:
        print('ERROR: prev file duration')
        prev_duration = file_clicks[0][1] - float(data_ordered[str(key)][0][2])
        
    print('prev file duration: ', prev_duration)
    ##############################################################
    

    audio_annot_with_labels = audio_annot.copy()
    for i in range(len(file_clicks)):
        whale_number, click_time = file_clicks[i]
        indices = time_to_indices(click_time = click_time - prev_duration)
        for idx in indices:
            audio_annot_with_labels[idx, 1].append(click_time - prev_duration)
    # print(audio_annot_with_labels[1264 : 1267])
    
    custom_annot_save_dir = main_dir  + 'click_regress_all_admin/' + audio_rootname + '_annot_with_global_times.p'
    pickle.dump(audio_annot_with_labels, open(custom_annot_save_dir, 'wb'))
        
    c = 0
    for i in range(audio_annot_with_labels.shape[0]):
        local_times = []
        global_times = audio_annot_with_labels[i, 1]
        c += len(global_times)
        for t in global_times:
            local_times.append(t - i)
        audio_annot_with_labels[i, 1] = local_times
    print('total of global times (with duplicates): ', c)       
    # print(audio_annot_with_labels[1264 : 1267])
    
    custom_annot_save_dir = main_dir  + 'click_regress_all_admin/' + audio_rootname + '_annot_with_local_times.p'
    pickle.dump(audio_annot_with_labels, open(custom_annot_save_dir, 'wb'))

