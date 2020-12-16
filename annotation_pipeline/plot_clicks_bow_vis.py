# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 06:29:59 2020

@author: jkakl

source: https://github.com/53RT/Highly-Connected-Subgraphs-Clustering-HCS

"""

import networkx as nx
import hcs

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import numpy as np
import os
from numpy import genfromtxt 
import pickle
import pandas as pd
from scipy import ndimage
import csv

from numpy import genfromtxt 




#################################################################################


def get_clicks_info(conv_clicks, idx2click):
    clicks_info = []
    for i in range(len(conv_clicks)):
        whale_number, click_time = conv_clicks[i]
        info = {"idx": i, "click_num": idx2click[i], "click_time": click_time, "whale_num": whale_number}
        clicks_info.append(info)
        
    if clicks_info == sorted(clicks_info, key=lambda click : click["click_time"]):
        print('clicks info is sorted based on click time')
    if clicks_info == sorted(clicks_info, key=lambda click : click["click_num"]):
        print('clicks info is sorted based on click number')
    
    return clicks_info
    


   
    
   
def get_ground_truth_grouping(list_files=None):
    actual_whale2clicks = {}
    actual_click2whale = {}
    for idx in range(len(list_files)):
        whale_num = int(list_files[idx][1])        
        # click_num = idx2click[idx]        
        actual_click2whale[idx] = whale_num
        # actual_click2whale[click_num] = whale_num
        if whale_num not in actual_whale2clicks.keys():
            actual_whale2clicks[whale_num] = [idx,]
            # actual_whale2clicks[whale_num] = [click_num,]
        else:
            actual_whale2clicks[whale_num].append(idx)
            # actual_whale2clicks[whale_num].append(click_num)
            
    return actual_click2whale, actual_whale2clicks



def check_dicts(click2whale, whale2clicks):
    for whale_num in whale2clicks:
        whale_clicks = [click_idx for click_idx in click2whale if click2whale[click_idx] == whale_num]
        if whale2clicks[whale_num] != whale_clicks:
            print('whale num: ', whale_num, ' dictionaries are inconsistent')




def print_grouping(click2whale, whale2clicks, idx2click):
    for whale_num in sorted(whale2clicks.keys()):
        clicks_this_whale = [idx2click[idx] for idx in whale2clicks[whale_num]]
        print('whale num: ', whale_num)
        print('total clicks: ', len(clicks_this_whale))
        print(clicks_this_whale, '\n')
        
    # for click_num in actual_click2whale:
    #     print('click num: ', click_num, actual_click2whale[click_num])
    print('---------------------------------------------------------------------------------------\n')


###########################################################################################


def plot_clustered_clicks_bow_vis(conv_clicks = None, actual_whale2clicks = None, actual_click2whale = None,
                          idx2click = None, clicks_info = None, all_codas = None, 
                          start_time = None, end_time = None, save_dir = None):
    
    if start_time == None or end_time == None:
        print('first click time: ', conv_clicks[0][1])
        print('last click time: ', conv_clicks[-1][1])        
        start_time = int(float(conv_clicks[0][1]))
        end_time = int((float(conv_clicks[-1][1]))) + 1      
        # print(type(start_time), type(end_time))
    
    print('--------------Plotting clustered clicks bow visualization--------------------')
    print('total number of clicks to plot: ', len(conv_clicks))
    print('start time: ', start_time)
    print('end time: ', end_time)
    
    plt.rcParams.update({'figure.max_open_warning': 0})
    
    markers = {0: "p", 1: ".", 2: "*", 3: "1", 4: "v", 5: "x", 6: "+", 7: "s", 8: ">", 9: "D"}
    colour_chart = ['r','r','g','b','y','c','m','#A2142F','#4DBEEE','#7E2F8E','#77AC30','#D95319','#0072BD']
    
    
    plt.clf()
    plt.gcf().set_size_inches(20, 10)
    
    
    for whale_num in all_codas.keys():
        whale_codas = all_codas[whale_num]
        for i in range(len(whale_codas)):
            coda_i = whale_codas[i]
            for j in range(len(coda_i)):
                x = coda_i[0]
                y = coda_i[j] - coda_i[0]
                print('x, y: ', x, y)
                plt.plot([x], [y], marker = markers[1], color = colour_chart[whale_num], markersize = 5)
        
    # plt.legend()
    plt.xlabel("Time in file")
    plt.ylabel("Time inside coda")
    
    plt.savefig('test_vis_bow_style_' + str(start_time) + '_' + str(end_time) + '.png')        
    # plt.show()
    plt.close()
    
    





def plot_clustered_clicks(conv_clicks = None, actual_whale2clicks = None, actual_click2whale = None,
                          pred_whale2clicks = None, pred_click2whale = None, idx2click = None, clicks_info = None,
                          start_time = None, end_time = None, every_num_sec = None, save_dir = None):
    
    if start_time == None or end_time == None:
        print('first click time: ', conv_clicks[0][1])
        print('last click time: ', conv_clicks[-1][1])        
        start_time = int((float(conv_clicks[0][1]) // every_num_sec) * every_num_sec)
        end_time = int((float(conv_clicks[-1][1]) // every_num_sec) * every_num_sec + every_num_sec)        
        # print(type(start_time), type(end_time))
    
    print('--------------Plotting clustered clicks--------------------')
    print('total number of clicks to plot: ', len(conv_clicks))
    print('start time: ', start_time)
    print('end time: ', end_time)
    print('every: ', every_num_sec, '\n')
    
    plt.rcParams.update({'figure.max_open_warning': 0})
    
    markers = {0: "p", 1: ".", 2: "*", 3: "1", 4: "v", 5: "x", 6: "+", 7: "s", 8: ">", 9: "D"}
    # color_set = {0: "grey", 1: "b", 2: "m", 3: "y", 4: "g", 5: "gold", 6: "c", 7: "violet", 8: "cornflowerblue", 9: "peru", 10: "olivedrab"}
    
    color_set = {0: "black", 1: "grey", 2: "m", 3: "y", 4: "c", 5: "b", 6: "g", 7: "violet", 8: "cornflowerblue", 9: "peru", 10: "olivedrab" }
    colour_chart = ['r','r','g','b','y','c','m','#A2142F','#4DBEEE','#7E2F8E','#77AC30','#D95319','#0072BD']
    
    def get_click_idxs_in_window(start=None, end=None):
        click_idxs_in_window = []
        for idx in range(len(conv_clicks)):
            if clicks_info[idx]["click_time"] >= start and clicks_info[idx]["click_time"] <= end:
                click_idxs_in_window.append(idx)
        return click_idxs_in_window
    
    
    num_all_clicks = 0
    total_clusters = len(pred_whale2clicks.keys())
    total_whales = len(actual_whale2clicks.keys())
    
    for t in range(start_time, end_time, every_num_sec):
        
        click_idxs_in_window = get_click_idxs_in_window(start = t, end = t + every_num_sec)
        num_all_clicks += len(click_idxs_in_window)
        
        if t % 20 == 0:
            print("num seconds: ", t)
        
        number_of_subplots = 2
        # fig, ax = plt.subplots(number_of_subplots,  1, sharex=True)
        fig, ax = plt.subplots(number_of_subplots,  1, sharex=False)
        
        ax[0].set_xlim([t, t + every_num_sec])
        ax[1].set_xlim([t, t + every_num_sec])       
        
        for idx in click_idxs_in_window:
            click_time = clicks_info[idx]["click_time"]
            cluster_num = pred_click2whale[idx]
            whale_num = actual_click2whale[idx]
            
            color_pred = 'black' if cluster_num not in color_set else color_set[cluster_num]
            color_gt = colour_chart[whale_num]
            
            ax[0].axvline(click_time, color = color_pred, ls='-', linewidth = 0.6)
            ax[1].axvline(click_time, color = color_gt, ls='-', linewidth = 0.6)
        
        ax[0].set_title("Predicted clustering: " + str(total_clusters) + " clusters")
        ax[1].set_title("Ground truth clustering: " + str(total_whales) + " whales")        
        
        plt.tight_layout()
        
        plt.savefig(save_dir + str(t) + '_' + str(t + every_num_sec) + '.png')        
        # plt.show()
        plt.close()    
    
    print('number of clicks plotted: ', num_all_clicks)    
    print('--------------------------------------------------')
    
    






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


def rootname_to_book_num(audio_rootname):
    for book_num in range(len(books)):
        book, filename = books[book_num]
        if audio_rootname == filename:
            return book_num
        

def get_codas_info(books, audio_rootname):
    
    book_num = rootname_to_book_num(audio_rootname)    
    book, filename = books[book_num]       
            
    print('Processing book ' + str(book_num) + ' filename ' + filename)
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
    
    info = []
    
    for i in range(num_stories):
        [whale_number, t_init, num_clicks, click_times, average_power, click_power] = parseCoda(book, i, time_origin)
        total_clicks += num_clicks
        
        info.append([whale_number, t_init, num_clicks, click_times])
        
        for j in range(num_clicks):
            click_time = t_init + click_times[j]
            # click = data[int(click_time*rate - 0.5*rate) : int(click_time*rate + 0.5*rate)]                
            
            # file_clicks.append((click, whale_number, click_time)) ## will need it for file_ordered
            file_clicks.append((whale_number, click_time))
            all_times.append(click_time)    
    
    # print('file clicks length: ', len(file_clicks))
    return info
    


def get_max_ici(codas_info):
    max_ici = 0
    max_i = -1
    max_j = -1
    sum_icis = 0
    num_icis = 0
    for i in range(len(codas_info)):
        [whale_number, t_init, num_clicks, click_times] = codas_info[i]
        for j in range(num_clicks - 1):
            sum_icis += click_times[j+1] - click_times[j]
            num_icis += 1
            if click_times[j+1] - click_times[j] > max_ici:
                max_ici = click_times[j+1] - click_times[j]
                max_i = i
                max_j = j
                
    return max_ici, max_i, max_j, sum_icis, num_icis


def get_min_time_diff_between_codas(codas_info):
    min_time_diff = 1_000
    coda_idx = -1
    
    for i in range(len(codas_info) - 1):
        
        [whale_number, t_init, num_clicks, click_times] = codas_info[i]
        if click_times != sorted(click_times):
            print('found unsorted click times!')
            
        # only care about same whale
        if codas_info[i][0] == codas_info[i+1][0]:
            last_click_time_i = codas_info[i][1] + codas_info[i][3][-1]
            first_click_time_ip1 = codas_info[i+1][1]
            
            if first_click_time_ip1 - last_click_time_i < min_time_diff:
                min_time_diff = first_click_time_ip1 - last_click_time_i
                coda_idx = i
                
    return min_time_diff, coda_idx
                
        

def group_clicks_to_codas(conv_clicks, whale2clicks, time_diff_thr):
    all_codas = {}
    for whale_num in whale2clicks.keys():
        click_idxs = whale2clicks[whale_num]
        click_times = []
        for idx in click_idxs:
            if conv_clicks[idx][0] != whale_num:
                print('incosistency: ', idx, conv_clicks[idx][0], whale_num)                
            click_time = conv_clicks[idx][1]
            click_times.append(click_time)
        
        whale_codas = []
        curr_coda = [click_times[0]]
        for i in range(1, len(click_times)):
            if click_times[i] - curr_coda[-1] <= time_diff_thr:
                curr_coda.append(click_times[i])
            else:
                ## aliasing?
                whale_codas.append(curr_coda)
                curr_coda = [click_times[i]]
        whale_codas.append(curr_coda)
        
        print('whale: ', whale_num)
        print('click times length: ', len(click_times))
        print('total clicks in codas: ', sum([len(coda) for coda in whale_codas]))
        print('whale ', whale_num, ' total codas: ', len(whale_codas))
        
        all_codas[whale_num] = whale_codas
    
    return all_codas
        


def check_file_clicks_against_list_files(file_clicks, list_files, prev_duration = 0):
    # print('file clicks: ', len(file_clicks))
    # print('list files: ', len(list_files))
    
    # print('file clicks is sorted: ', file_clicks == sorted(file_clicks, key=lambda pair : pair[1]))
    # print('list files is sorted on time: ', list_files == sorted(list_files, key=lambda triple : float(triple[2])))
    # print('list files is sorted on ext: ', list_files == sorted(list_files, key = lambda triple: int(triple[0].split('/')[-1][10:16])))          
    all_good = True
    for i in range(len(file_clicks)):        
        # (click_audio_dir, whale_num, click_time) = list_files[i]
        if float(list_files[i][2]) + prev_duration != float(file_clicks[i][1]):
            print('issue: ', i, file_clicks[i][1], list_files[i][2])
            all_good = False
            # break
    if all_good:
        print('checked file_clicks against list_files: all good')



def filter_list_files(list_files, file_clicks, prev_duration = 0):
    file_clicks_times = set(pair[1] for pair in file_clicks)
    # print('file clicks all times: ', len(file_clicks_times))
    new_list_files = []
    for i in range(len(list_files)):
        (click_audio_dir, whale_num, click_time) = list_files[i]
        
        if float(click_time) + prev_duration in file_clicks_times:
            new_list_files.append((click_audio_dir, whale_num, click_time))            
    return new_list_files
        
        
    
    



if __name__ == '__main__': 
    
    audio_rootname = 'sw061b001' ## year 2015
    # audio_rootname = 'sw061b002' ## year 2015
    
    key = 0
    start = 0
    end = 235    
    # start = 50
    # end = 110
    # start = 100
    # end = 140    
    # start = 235
    # end = 399
    
    ## first 5 minutes
    # start = 0
    # end = 399
    
    # start = 0
    # end = 1302
    
    
    # key = 1
    # start = 559
    # end = 747    
    ## best - three whales
    # start = 747
    # end = 844
    
    # start = 0
    # end = 1683

    
    print('\n')
    print('---------------------' + audio_rootname + '---------------------\n')   
        
    home_dir = '/data/vision/torralba/scratch/ioannis/clustering/'
    labels_dir = home_dir + 'click_regress_all_admin/' + audio_rootname + '_labels.p'    
    file_clicks = pickle.load(open(labels_dir, 'rb'))    
    
    # data_ordered_dir = '/data/scratch/ioannis/click_seperator/correct_file_ordered.p'
    data_ordered_dir = '/data/vision/torralba/scratch/ioannis/clustering/click_separator_training/file_ordered_correct_times.p'
    data_ordered = pickle.load(open(data_ordered_dir, 'rb'))
    
    #############################################################
    ## VERY CUSTOM WAY TO GET duration of sw061b001, works for sw061b001, sw061b002
    prev_duration = file_clicks[0][1] - float(data_ordered[str(key)][0][2])
    # print('prev file duration: ', prev_duration)
    ##############################################################
    
    ## list_files is a list of tuples (click_audio_dir, whale_num, click_time) ##
    # list_files = data_ordered[str(key)][start : end]
    list_files = data_ordered[str(key)]    
    list_files = filter_list_files(list_files, file_clicks, prev_duration)
    list_files = list_files[start : end]
    print('filtered list files: ', len(list_files), len(set(list_files)))
    
        
    actual_click2whale, actual_whale2clicks = get_ground_truth_grouping(list_files)
    check_dicts(actual_click2whale, actual_whale2clicks)    
    
    conv_clicks = file_clicks[start : end]
    
    idx2click = {idx : start + idx  for idx in range(len(conv_clicks))}
    # whale_number, click_time = conv_clicks[i]
    
    print('Ground truth grouping: \n')
    print_grouping(actual_click2whale, actual_whale2clicks, idx2click)

    
    # check_file_clicks_against_list_files(file_clicks, list_files)
    check_file_clicks_against_list_files(conv_clicks, list_files, prev_duration)
    
    clicks_info = get_clicks_info(conv_clicks, idx2click)
    
    
    #####################################################################################################
    
    
    file_location = '/data/scratch/ioannis/dataset/coda_parsing_with_metadata.csv'
    # file_location = '/data/scratch/ioannis/click_seperator/new_coda_metadata.csv'
    
    my_data = genfromtxt(file_location, delimiter=',',dtype=None,encoding="utf8")
    print('my_data shape: ', my_data.shape)
    
    annotations_dir = '/data/scratch/ioannis/click_seperator/list_annotations.p'
    books = pickle.load(open(annotations_dir, 'rb'))
    
    # home_dir = '/data/scratch/ioannis/'
    home_dir = '/data/vision/torralba/scratch/ioannis/clustering/'
    
    
    codas_info = get_codas_info(books, audio_rootname)
    # codas_info = codas_info[:54] ## clicks 0 - 398
    codas_info = codas_info[:39] ## clicks 0 - 234
    
    print('------codas info-------')
    print('codas sorted: ', codas_info == sorted(codas_info, key = lambda coda : coda[1]))
    # print(codas_info[20:25])
    
    for whale_num in actual_whale2clicks:
        print('whale ', whale_num, ' total codas: ', len([i for i in range(len(codas_info)) if codas_info[i][0] == whale_num]))
    
    max_ici, max_i, max_j, sum_icis, num_icis = get_max_ici(codas_info)
    
    print('max ici: ', max_ici)
    # print('indices: ', max_i, max_j)
    # print(codas_info[max_i])
    
    # print('num icis: ', num_icis)
    # print('avg ici: ', sum_icis / num_icis)
    
    min_time_diff, coda_idx = get_min_time_diff_between_codas(codas_info)
    
    print('min time diff between codas: ', min_time_diff)
    # print('coda idx: ', coda_idx)
    # print(codas_info[coda_idx])
    # print(codas_info[coda_idx + 1])
    
    # for i in range(len(codas_info)):
    #     print(i, codas_info[i])
    print('----------------------------------')
    
    print('------grouping clicks to codas---------')
    time_diff_thr = 0.7
    all_codas = group_clicks_to_codas(conv_clicks, actual_whale2clicks, time_diff_thr)    
    print('all codas: ', sum(len(all_codas[whale_num]) for whale_num in all_codas.keys()))
    print('----------------------------------')
    
    
    print('--------Plotting codas bow vis style------\n')
    # plot_clustered_clicks_bow_vis(conv_clicks = conv_clicks, all_codas = all_codas)
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    


