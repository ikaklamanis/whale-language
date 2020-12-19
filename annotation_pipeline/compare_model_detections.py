# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 05:44:30 2020

@author: jkakl
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 03:36:08 2020

@author: jkakl
"""

import numpy as np
import io, os
import pickle
import argparse
import math
import shutil
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sn
import pandas as pd
import librosa
import librosa.display
import cv2
import random
from collections import Counter

import sys
import pylab as pl
from sklearn.metrics import precision_recall_curve


######### Some constants ############

min_timestep = 0
max_timestep = 22050
window_size = 2000
black_window_size = 150

#####################################



def get_year(file_path):
    yr = int(file_path.split('/')[-3])
    return yr
    
def get_rootfile(file_path):
    root = str(file_path.split('/')[-2])
    return root

def get_ext(file_path):
    file_ext = str(file_path.split('/')[-1][:-4])
    return file_ext




def get_predictions(unannot_preds_files = None, in_order = False):
    predictions = [] ## not dict because there are duplicates
    idx = 0
    for pred_file in unannot_preds_files:
        info = pickle.load(open(pred_file, 'rb'))
        audio_dir, data, labels = info
        # print(data.shape)
        preds = data[0,:].tolist()
        predictions.append((audio_dir, labels, preds))
        idx +=1
    
    if in_order:
        ## sort based on file number (000001, 000256, etc.)
        predictions.sort(key=lambda triple: triple[0][1].split('/')[-1][10:16])
    
    return predictions



def in_close_range(positives_clustered={}, pred_i=0, range_size=30):
    pred_j = None
    for pos in positives_clustered.keys():
        if abs(pos - pred_i) <= range_size:
            pred_j = pos
            break
    return pred_j
            
        
## can also work with positives across multiple 1-sec files
def average_close_positives(positives=[], range_size=30):
    
    positives_clustered = {}
    for i in range(len(positives)):
        pred_i = positives[i]
        if pred_i not in positives_clustered.keys():
            pred_j = in_close_range(positives_clustered, pred_i, range_size)
            if pred_j != None:
                positives_clustered[pred_j].append(pred_i)
            else:
                positives_clustered[pred_i] = [pred_i,]        
        elif pred_i in positives_clustered.keys():
            positives_clustered[pred_i].append(pred_i)
    
    positives_averaged = []
    for key in sorted(positives_clustered.keys()):
        close_positives = positives_clustered[key]
        avg_positive = int(sum(close_positives) / len(close_positives))
        positives_averaged.append(avg_positive)
    
    return positives_averaged
  





def align_preds_with_labels(predictions = None, annot_global_times = None, conf_number = 15, 
                            average_preds = True, range_size = 25):
        
    print('CONF THRESHOLD: ', conf_number)
    print('avg range: ', range_size)
    
    all_positives = []
    same_len = 0
    diff_len = 0

    only_pred = 0
    one_sec_files_preds_no_gt = []
    
    only_label = 0
    total = 0
    neither = 0
    
    all_labels = []
    
    raw_preds_number = 0
    avg_preds_number = 0
    
    ## predictions list is sorted on audio dir (000000, 000001, etc.)
    for i in range(len(predictions)):
        
        ## labels_empty is empty list [] fed to model as "labels" in test mode
        audio_dir, labels_empty, preds = predictions[i] ## here audio_dir is a tuple of 3 audio dirs: (prev, curr, next)
        
        idx = int(audio_dir[1].split('/')[-1][10:16])
        # labels = sw061b001_annot_local_times[idx, 1]
        labels = annot_global_times[idx, 1]
        # labels = [ l * 22050 for l in labels] ## they were in seconds
        
        
        audio_dir = audio_dir[1] ## middle audio directory    
        # audio, rate = librosa.load(audio_dir, mono=False)
        audio_shape1 = 22050 + 2000 # 24050
        
        
        positives = []
        for k in range(min_timestep, audio_shape1 - window_size):
            if preds[k] > 0:
                actual_time = k + window_size/2 - black_window_size/2 + preds[k]
                positives.append(actual_time)
        
        ## Threshold / Confidence idea
        c = Counter(positives)
        conf_positives = [pos for pos in positives if c[pos] > conf_number]
        # pos_sorted = [pos/rate for pos in sorted(list(set(conf_positives)))]
        pos_sorted = [pos for pos in sorted(list(set(conf_positives)))]
        
        
        raw_preds_number += len(pos_sorted)
        if average_preds:
            pos_sorted_averaged = average_close_positives(pos_sorted, range_size)
            
            avg_preds_number += len(pos_sorted_averaged)
            pos_sorted = pos_sorted_averaged
    
        
        if len(pos_sorted) == 0 and len(labels) == 0:
            neither += 1            
        elif len(pos_sorted) > 0 and len(labels) == 0:
            only_pred += 1
            one_sec_files_preds_no_gt.append((audio_dir, pos_sorted)) ## [(audio dir, positive preds),]            
        elif len(pos_sorted) == 0 and len(labels) > 0:
            only_label += 1
            # print(idx)        
        if len(pos_sorted) > 0 and len(labels) > 0:
            if len(pos_sorted) == len(labels):
                same_len += 1
            else:
                diff_len += 1    
        total += 1    
    
        
        global_positives = []
        ## pos_sorted has frame numbers, i.e. 500, 6094, 19084
        for pos in pos_sorted:
            # print(pos_sorted)
            global_click_time_pred = idx + pos/22050
            global_positives.append(global_click_time_pred)
            
            
        all_positives += global_positives    
        all_labels += labels
    
    
    print('raw preds total: ', raw_preds_number)
    print('averaged preds total: ', avg_preds_number)
    
    print(len(set(all_positives)), len(all_positives))        
        
    return all_labels, all_positives, one_sec_files_preds_no_gt



############################# MATCH PREDICTIONS WITH LABELS ###############################


###### Attention: matches clicks to preds, but may reuse a prediction ######

def match_labels_with_preds(all_positives = None, all_labels = None, epsilon = 100, reuse_preds = True):
    print("--------Matching labels with preds---------------")
    
    print('epsilon: ', epsilon)
    print('reuse preds: ', reuse_preds)
    
    ## they remain sorted, no duplicates
    all_preds_positives = [pred * 22050 for pred in all_positives]
    all_labels_positives = [lab * 22050 for lab in all_labels]
    
    number_of_matches = 0
    total_labels = 0
    duplicate_match_freqs = {}
    
    missed_clicks = []
    matched_clicks = []
    
    matched_preds = set()
    # unmatched_preds = []
    
    for i in range(len(all_labels_positives)):
        label_i = all_labels_positives[i]
        match = False
        duplicate_matches = 0
        for j in range(len(all_preds_positives)):
            pred_j = all_preds_positives[j]
            if abs(label_i - pred_j) <= epsilon:
                
                if reuse_preds:
                    match = True
                    duplicate_matches += 1                    
                    if pred_j not in matched_preds:
                        matched_preds.add(pred_j)
                else:
                    if pred_j not in matched_preds:
                        matched_preds.add(pred_j)
                        match = True
                        duplicate_matches += 1               
                # break
            
        if match:
            matched_clicks.append(label_i)
            number_of_matches += 1
        else:
            missed_clicks.append(label_i)
            
        if duplicate_matches not in duplicate_match_freqs:
            duplicate_match_freqs[duplicate_matches] = 1
        else:
            duplicate_match_freqs[duplicate_matches] += 1
        
        total_labels += 1
    
    unmatched_preds = sorted([pred_pos for pred_pos in all_preds_positives if pred_pos not in matched_preds])
    
    missed_clicks = sorted([label_pos / 22050 for label_pos in missed_clicks])
    matched_clicks = sorted([label_pos / 22050 for label_pos in matched_clicks])
    
    unmatched_preds = sorted([pred_pos / 22050 for pred_pos in unmatched_preds])
    matched_preds = sorted([pred_pos / 22050 for pred_pos in matched_preds])
    
    print('Matched clicks: ', number_of_matches, len(matched_clicks))
    # print('Missed clicks: ', len(missed_clicks))
    # print('total_clicks: ', total_labels)
    print('match percentage: ', 100 * number_of_matches / len(all_labels_positives))
    print('duplicate matches frequencies: ', duplicate_match_freqs)
    
    # print('Matched preds: ', len(matched_preds))
    # print('Unmatched preds: ', len(unmatched_preds))
    
    print('--------------------------------------------------')    
    
    return missed_clicks, matched_clicks, unmatched_preds, matched_preds

        
        
        
#############################    PLOT    ################################

def plot_preds_with_labels(all_positives, all_labels, missed_clicks, start_time=0, end_time=300, every_num_sec=40, 
                           save_dir=None, plot_missed_clicks=True):
    
    print('--------------Plotting preds vs labels--------------------')
    print('start time: ', start_time)
    print('end time: ', end_time)
    print('every: ', every_num_sec, '\n')
    
    
    num_all_labels = 0
    
    for i in range(start_time, end_time, every_num_sec):
        
        if i % 1000 == 0:
            print("num seconds: ", i)
        
        number_of_subplots = 2
        # fig, ax = plt.subplots(number_of_subplots,  1, sharex=True)
        fig, ax = plt.subplots(number_of_subplots,  1, sharex=False)
        
        ax[0].set_xlim([i, i + every_num_sec])
        ax[1].set_xlim([i, i + every_num_sec])
        
        these_positives = [pos for pos in all_positives if pos >= i and pos <= i + every_num_sec]
        for t in these_positives:
            ax[0].axvline(t, color='b', ls='-', linewidth = 0.3)
        
        these_labels = [lab for lab in all_labels if lab >= i and lab <= i + every_num_sec]
        for t in these_labels:
            ax[1].axvline(t, color='g', ls='-', linewidth = 0.3)
            
        num_all_labels += len(these_labels)
        
        ### plot red lines on missed clicks
        if plot_missed_clicks:
            for click_time in missed_clicks:
                if click_time >= i and click_time <= i + every_num_sec:
                    ax[1].axvline(click_time, color='r', ls='-', linewidth = 0.4)       
        
        # ax[0].set_axis_off()
        
        ax[0].set_title("Predictions")
        ax[1].set_title("Ground truth")        
        
        plt.savefig(save_dir + str(i) + '_' + str(i + every_num_sec) + '.png')        
        # plt.show()
        plt.close()    
    
    print('number of labels plotted: ', num_all_labels)    
    print('--------------------------------------------------')




        
        


def create_idx_to_preds_dicts(predictions = None, matched_preds = None, unmatched_preds = None):     
    idx_to_matched_preds = {idx : [] for idx in range(len(predictions))}
    idx_to_unmatched_preds = {idx : [] for idx in range(len(predictions))}
    for pred in matched_preds:
        if int(pred) in idx_to_matched_preds:
            idx_to_matched_preds[int(pred)].append(pred - int(pred))
    for pred in unmatched_preds:
        if int(pred) in idx_to_unmatched_preds:
            idx_to_unmatched_preds[int(pred)].append(pred - int(pred))
    
    # print('idx window to preds: ', len(idx_window_to_preds.keys()))
    # print('total preds: ', sum(len(idx_window_to_preds[idx]) for idx in idx_window_to_preds.keys()))
    return idx_to_matched_preds, idx_to_unmatched_preds


    
    
def plot_missed_clicks_waveforms(predictions = None, missed_clicks=None, plot_preds=False, plot_labels = False,
                                 idx_to_matched_preds = None, idx_to_unmatched_preds = None, 
                                 missed_clicks_waveforms_dir = None):    
    num_1_sec_files_plotted = 0
    
    for i in range(len(predictions)):
        
        audio_dir = predictions[i][0][1] ## recall: tuple of 3 audio dirs: (prev, curr, next)
        idx = int(audio_dir.split('/')[-1][10:16])
        
        missed_clicks_this_window = []
        for j in range(len(missed_clicks)):
            click_time = missed_clicks[j]
            if idx <= click_time and click_time <= idx + 1:
                missed_clicks_this_window.append(click_time)
        
        
        if len(missed_clicks_this_window) > 0:
            
            num_1_sec_files_plotted += 1
            
            audio, rate = librosa.load(audio_dir, mono=False) ## middle file path
            data_r = audio[1,:]
            
            ext  = audio_dir.split('/')[-1][:-4]
            
            year = get_year(audio_dir)
            root_file = get_rootfile(audio_dir)
        
            mytime = np.arange(0, len(data_r)) / rate
            # Plot audio over time
            fig = plt.figure(frameon=False)
            
            if plot_labels:
                for click_time in sorted(list(set(missed_clicks_this_window))):
                    # print(click_time, click_time - idx)
                    plt.axvline(click_time - idx, color='red', ls='-', linewidth=1.5)
            
            number_of_preds_plotted = 0
            
            if plot_preds:
                for x_pred in idx_to_matched_preds[idx]:
                    plt.axvline(x = x_pred, color='green', ls='-', linewidth=1.5)
                    number_of_preds_plotted += 1

                for x_pred in idx_to_unmatched_preds[idx]:
                    plt.axvline(x = x_pred, color='black', ls='--', linewidth=1.5)
                    number_of_preds_plotted += 1
                    
                    
            plt.plot(mytime, data_r, 'b:')
            
            custom_lines = [Line2D([0], [0], color='red', ls='-', lw=1.5),
                            Line2D([0], [0], color='green', ls='-', lw=1.5),
                            Line2D([0], [0], color='black', ls='--', lw=1.5)]
            plt.legend(custom_lines, ['Missed labels', 'Matched preds', 'Unmatched preds'])
            
            plt.title(ext)
                        
            plt.savefig(missed_clicks_waveforms_dir + ext)
            plt.close()
            
    print('number of 1-sec files plotted: ', num_1_sec_files_plotted)
            
                
            
    

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



def click_detector_parse_entire_file(audio_rootname = None, model_preds_dir = None, conf_number = 15, model_version = None):
    '''
    
    Main method that parses entire audio file based on (relocated) detections of click detector. 
    Makes use of helper methods above. Four main steps: 
    
    1. Parse, threshold, average detected click times, and "align" them with ground truth click times: 
     - conf_number: confidence threshold number to use to accept detections (based on frequency of detection) 
                    Do not accept times t if they appear in the detected times less than 'conf_number' times.
                    Good value might be around 60. Can be passed a argument
     - average_preds (boolean): if true, average detections. Currently not passed as a argument
     - range_size: size of window to use when averaging detections, measured in # frames (1 sec = 22050 frames)
                 default value: 25, ~ 1 millisecond. Currently not passed as a argument
    
        ** Effect: saves processed detected times in pickle file for later use.  
    
                 
    2. Compare and "match" detections with ground truth click times:
         - epsilon: max time distance (in #frames) to use in order to match a ground truth click time with a detected time
                    i.e., click time t (sec) is matched if there exists a detected click time pred_t (sec) such that:
                        abs(t - pred_t) <= epsilon / rate, where rate is 22050
                    default value: 100 (frames), ~ 3 milliseconds. Currently not passed as a argument
                    
         - reuse_preds (boolean): If true, then can use same detection more than once when trying to match ground truth click times.
                                  If false, each detection can be used at most once when matching click times
                                  default value: false. Currently not passed as a argument
                                  
        Returns: lists of missed_clicks, matched_clicks, unmatched_preds, matched_preds
        **** currently this data is not being saved for later use ****    
    
    3. Plot detected click times (top subplot) vs ground truth click times (bottom subplot)
      - start_time = 0
      - end_time = duration of audio_rootname
      - every_num_sec (default = 100): length of time window for each plot
      - save_dir: specified in code
      ** These are not currently passed as arguments
     
    4. Plot waveforms of missed clicks
     - save_dir: specified in code. Currently not passed as an argument


    *** NOTE: "Currently not passed as a parameter" means that currently this parameter is not an argument for this method,
               but its value is specified instead within the method's body.
    '''
    
    
    def rootname_to_book_num(audio_rootname, books):
        for book_num in range(len(books)):
            book, filename = books[book_num]
            if audio_rootname == filename:
                return book_num
            
    annotations_dir = '/data/scratch/ioannis/click_seperator/list_annotations.p'
    books = pickle.load(open(annotations_dir, 'rb'))    
    key = rootname_to_book_num(audio_rootname, books)
    
    dataset_directory = '/data/scratch/ioannis/dataset/'
    file_directory = get_directory(dataset_directory, audio_rootname)
    year = get_year(file_directory)
    
    
    print('----------Processing audio rootname: ', audio_rootname, '---------------- \n')
    print('---------------model version: ', model_version)
        
    new_detector_dir = '/data/vision/torralba/scratch/ioannis/click_regress/training/detector_noise_right_edges_annot_data/'
    noise_visual_dir = new_detector_dir + 'models_trained_with_noise_detections_visuals/' + model_version + '/'    
    if not os.path.exists(noise_visual_dir):
        os.makedirs(noise_visual_dir)       
        
    custom_dir = model_preds_dir + str(year) + '/' + audio_rootname + '/'
    unannot_preds_files = [custom_dir + pred_file for pred_file in os.listdir(custom_dir)] ## very custom
    
    main_dir = '/data/vision/torralba/scratch/ioannis/clustering/'
    annot_global_times_dir = main_dir + 'click_regress_all_admin/' + audio_rootname + '_annot_with_global_times.p'
    annot_global_times = pickle.load(open(annot_global_times_dir, 'rb'))    
    ####################################
    labels_dir = main_dir + 'click_regress_all_admin/' + audio_rootname + '_labels.p'    
    file_clicks = pickle.load(open(labels_dir, 'rb'))    
    ###################################################################################################    
    
    predictions = get_predictions(unannot_preds_files = unannot_preds_files, in_order = True)
    
    conf_number = conf_number
    average_preds = True
    range_size = 25 ## 22 ~ 22050 / 1000
    ##########################################################################################################
    all_labels, all_positives, one_sec_files_preds_no_gt = align_preds_with_labels(predictions = predictions,
                                        annot_global_times = annot_global_times, conf_number = conf_number,
                                        average_preds = average_preds, range_size = range_size)
    ##########################################################################################################
    all_labels = sorted(set(all_labels))
    all_positives = sorted(set(all_positives))    
    print('all labels: ', len(all_labels))
    print('all positives: ', len(all_positives))
    
    all_positives_save_dir = noise_visual_dir + audio_rootname + '_all_positives_conf_' + str(conf_number) + '.p'
    pickle.dump(all_positives, open(all_positives_save_dir, 'wb')) ## USE THIS 
    
    print('---------------------------------------------------')
    
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
    
    
    epsilon = 100
    reuse_preds = False
    ##########################################################################################################
    missed_clicks, matched_clicks, unmatched_preds, matched_preds = match_labels_with_preds(all_positives = all_positives,
                                            all_labels = all_labels, epsilon = epsilon, reuse_preds = reuse_preds)
    ##########################################################################################################
    
    # matched_clicks_save_dir = noise_visual_dir + audio_rootname + '_matched_clicks_conf_' + str(conf_number) + '.p'
    # pickle.dump(matched_clicks, open(matched_clicks_save_dir, 'wb')) ## USE THIS
    # missed_clicks_save_dir = noise_visual_dir + audio_rootname + '_missed_clicks_conf_' + str(conf_number) + '.p'
    # pickle.dump(missed_clicks, open(missed_clicks_save_dir, 'wb')) ## USE THIS
    
    print('---------------------------------------------------')
    
    start_time = 0
    end_time = len(predictions)
    every_num_sec = 100    
    
    preds_vs_labels_plot_dir = noise_visual_dir + 'click_detector_' + audio_rootname + '_preds_vs_labels_every_' + str(every_num_sec) + '/'    
    if not os.path.exists(preds_vs_labels_plot_dir):
        os.makedirs(preds_vs_labels_plot_dir)    
    plot_missed_clicks = True    
    ##########################################################################################################
    plot_preds_with_labels(all_positives, all_labels, missed_clicks, start_time, end_time, every_num_sec, 
                            save_dir = preds_vs_labels_plot_dir, plot_missed_clicks = plot_missed_clicks)
    ##########################################################################################################
    
    print('---------------------------------------------------')
    
    idx_to_matched_preds, idx_to_unmatched_preds = create_idx_to_preds_dicts(predictions = predictions, 
                                       matched_preds = matched_preds, unmatched_preds = unmatched_preds)
    
    missed_clicks_waveforms_dir = noise_visual_dir + audio_rootname + '_missed_clicks_waveforms/'
    if not os.path.exists(missed_clicks_waveforms_dir):
        os.makedirs(missed_clicks_waveforms_dir)        
    ##########################################################################################################
    plot_missed_clicks_waveforms(predictions = predictions, missed_clicks = missed_clicks, 
                                  plot_preds = True, plot_labels = False, idx_to_matched_preds = idx_to_matched_preds, 
                                  idx_to_unmatched_preds = idx_to_unmatched_preds, 
                                  missed_clicks_waveforms_dir = missed_clicks_waveforms_dir)
    ########################################################################################################

    
    return all_positives    
    







