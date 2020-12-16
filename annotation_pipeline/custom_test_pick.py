# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 04:39:25 2020

@author: jkakl
"""

import numpy as np
import pickle
import math
import librosa
import librosa.display
from scipy.io import wavfile
import io, os
import argparse


    
    
def get_det_annot_data_in_range(audio_rootname, model_version, start, end):
    
    def rootname_to_book_num(audio_rootname, books):
        for book_num in range(len(books)):
            book, filename = books[book_num]
            if audio_rootname == filename:
                return book_num
            
    annotations_dir = '/data/scratch/ioannis/click_seperator/list_annotations.p'
    books = pickle.load(open(annotations_dir, 'rb'))    
    key = rootname_to_book_num(audio_rootname, books)
    
    
    main_dir = '/data/vision/torralba/scratch/ioannis/clustering/'
    
    new_detector_dir = '/data/vision/torralba/scratch/ioannis/click_regress/training/detector_noise_right_edges_annot_data/'
    noise_visual_dir = new_detector_dir + 'models_trained_with_noise_detections_visuals/' + model_version + '/'
    # detected_clicks_annot_dir = main_dir + 'click_regress_all_admin/detected_clicks_annot_data_' + audio_rootname + '.p'
    detected_clicks_annot_dir = noise_visual_dir + 'detected_clicks_annot_data_' + audio_rootname + '.p'    
    detected_clicks_annot = pickle.load(open(detected_clicks_annot_dir, 'rb'))
    print('all detected clicks annot data: ', detected_clicks_annot.shape)
    
    # labels_dir = '/data/scratch/ioannis/' + 'click_regress_all_admin/' + audio_rootname + '_labels.p'
    labels_dir = main_dir + 'click_regress_all_admin/' + audio_rootname + '_labels.p'
    file_clicks = pickle.load(open(labels_dir, 'rb'))
    
    data_ordered_dir = '/data/vision/torralba/scratch/ioannis/clustering/click_separator_training/file_ordered_correct_times.p'
    data_ordered = pickle.load(open(data_ordered_dir, 'rb'))    
    #############################################################
    ## VERY CUSTOM WAY TO GET duration of sw061b001, works for sw061b001, sw061b002
    # prev_duration = file_clicks[0][1] - float(data_ordered[str(key)][0][2])
    # print('prev file duration: ', prev_duration)   
    
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
    
    print('file clicks: ')
    print(file_clicks[start])
    print(file_clicks[end-1])
    
    start_time = int(file_clicks[start][1] - prev_duration)
    end_time = int(file_clicks[end-1][1] - prev_duration) + 1    
    print('extracting detections in range: Clicks ', start, '-', end-1)
    print('start time: ', start_time)
    print('end time: ', end_time)
    
    det_annot_data_in_range = []
    
    for i in range(detected_clicks_annot.shape[0]):
        crop_dir, click_time = detected_clicks_annot[i]
        if start_time <= click_time and click_time <= end_time:
            det_annot_data_in_range.append((crop_dir, click_time))
    
    # det_annot_data_in_range = np.array(det_annot_data_in_range, dtype=object)   
    return det_annot_data_in_range
    
    


def create_custom_data_in_from_detections(audio_rootname, det_annot_data, start = 0, end = 0, save_dir = None):
    
    def rootname_to_book_num(audio_rootname, books):
        for book_num in range(len(books)):
            book, filename = books[book_num]
            if audio_rootname == filename:
                return book_num
            
    annotations_dir = '/data/scratch/ioannis/click_seperator/list_annotations.p'
    books = pickle.load(open(annotations_dir, 'rb'))    
    key = rootname_to_book_num(audio_rootname, books)
    
    # # labels_dir = '/data/scratch/ioannis/' + 'click_regress_all_admin/' + audio_rootname + '_labels.p'
    # labels_dir = main_dir + 'click_regress_all_admin/' + audio_rootname + '_labels.p'
    # file_clicks = pickle.load(open(labels_dir, 'rb'))
    
    print('key: ', key)
    print('start: ', start)
    print('end: ', end)
    
    print('detected clicks annot data:')
    print('start time: ', det_annot_data[0][1])
    print('end time: ', det_annot_data[-1][1])    

    
    ## find total number of clicks
        
    total_clicks = len(det_annot_data)        
    print('total num of detected clicks: ', total_clicks)
    ## create numpy array data_in
    num_all_pairs = int(total_clicks * (total_clicks - 1) / 2)
    # print(num_all_pairs)
    num_entries = 8
    data_in = np.array([['0' for j in range(num_entries)] for i in range(num_all_pairs)], dtype=object)
    idx = 0
        
    for i in range(len(det_annot_data)):
        # filename_1, label_1, click_time_1 = list_files[i]
        filename_1, click_time_1 = det_annot_data[i]
        # print('filename: ', filename_1.split('/')[-1])
        
        for j in range(i+1, len(det_annot_data)):
            # filename_2, label_2, click_time_2 = list_files[j]
            filename_2, click_time_2 = det_annot_data[j]
            
            ## TODO: find actual labels
            label_1 = -1
            label_2 = -1
            
            data_in[idx, :] = [filename_1, str(key), str(label_1), str(click_time_1),
                               filename_2, str(key), str(label_2), str(click_time_2)]            
            idx += 1
            
    print('custom_data_in: ', data_in.shape)
    # print(data_in[0, :])
    # print(data_in[-1, :])
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    custom_data_in_dir = save_dir + audio_rootname + '_clicks_' + str(start) + '_' + str(end) + '.p'
    pickle.dump(data_in, open(custom_data_in_dir, 'wb'))
    
    return data_in
    





