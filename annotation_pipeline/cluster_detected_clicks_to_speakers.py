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




################################################################################


def get_clicks_info_with_detections(conv_clicks, idx2click, idx2click_gt, click_idx_to_det_idx, det_click_times):
    clicks_info = []
    for i in range(len(conv_clicks)):
        whale_number, click_time = conv_clicks[i]
        
        if i in click_idx_to_det_idx:
            det_idx = click_idx_to_det_idx[i]
            det_time = det_click_times[det_idx]
        else:
            det_idx = None
            det_time = None
        
        info = {"idx": i, "click_num": idx2click_gt[i], "click_time": click_time, "whale_num": whale_number,
                "det_idx": det_idx, "det_time": det_time}
        clicks_info.append(info)
        
    if clicks_info == sorted(clicks_info, key=lambda click : click["click_time"]):
        print('clicks info is sorted based on click time')
    if clicks_info == sorted(clicks_info, key=lambda click : click["click_num"]):
        print('clicks info is sorted based on click number')
    
    return clicks_info


def augment_clicks_info(clicks_info_with_det, click_idx_to_det_idx, pred_click2whale, actual_click2whale):
    for click_idx in range(len(clicks_info_with_det)):
        if click_idx != clicks_info_with_det[click_idx]["idx"]:
            print('incorrect index')
        if clicks_info_with_det[click_idx]["whale_num"] != actual_click2whale[click_idx]:
            print('incorrect whale num')
            
        if click_idx in click_idx_to_det_idx:
            det_idx = click_idx_to_det_idx[click_idx]
            det_cluster = pred_click2whale[det_idx]
            clicks_info_with_det[click_idx]["det_cluster"] = det_cluster
        else:
            clicks_info_with_det[click_idx]["det_cluster"] = None
            print(click_idx, 'not detected')
            
    return clicks_info_with_det

    


def create_click_graph_from_detected_preds(comparisons_matrix, idx2click, det_click_times):
    G = nx.Graph()
    
    nodes = []
    for i in range(len(det_click_times)):
        det_time = det_click_times[i]
        nodes.append((i, {"idx": i, "det_click_num": idx2click[i], "det_click_time": det_time}))
    
    G.add_nodes_from(nodes)
    
    for u, data_u in G.nodes(data=True):
        for v, data_v in G.nodes(data=True):
            if data_u["idx"] != data_v["idx"]:
                ## if click u and click v are predicted to be from the same whale, add an edge (u,v)
                if comparisons_matrix[data_u["idx"], data_v["idx"]] == 1:
                    G.add_edge(u, v)
    
    return G.to_undirected()




def check_graph_against_info_with_detections(G, clicks_info_with_det, click_idx_to_det_idx, det_idx_to_click_idx):
    for u, data_u in G.nodes(data=True):
        if u in det_idx_to_click_idx:
            click_idx = det_idx_to_click_idx[u]
            if u != clicks_info_with_det[click_idx]["det_idx"] or data_u["idx"] != clicks_info_with_det[click_idx]["det_idx"]:
                print('incorrect index')
            if data_u["det_click_time"] != clicks_info_with_det[click_idx]["det_time"]:
                print('incorrect click time!')




def check_all_pairs(pair2comp, first_click_num, last_click_num):
    keys_set = set() 
    for click_1 in range(first_click_num, last_click_num + 1):
        for click_2 in range(first_click_num, last_click_num + 1):
            if click_1 != click_2:
                key = tuple(sorted([click_1, click_2]))
                keys_set.add(key)
                if key not in pair2comp.keys():
                    print(key, " is not a key")
    if keys_set == set(pair2comp.keys()): 
        # print("All keys included")
        return
    else:
        print("key sets are not the same")
        

def check_comparisons_matrix(comparisons_matrix):
    for i in range(comparisons_matrix.shape[0]):
        for j in range(comparisons_matrix.shape[1]):
            if comparisons_matrix[i,j] != comparisons_matrix[j,i]:
                print('Error: comparison matrix is inconsistent!')
   

def get_comparisons_matrix(preds=None):
    '''
    Returns
    -------
    comparisons_matrix : np array of shape (num_clicks, num_clicks) where 
                         entry [i,j] stores the prediction same/diff whale
                         when comparing click i, click j, for i != j.
                         The values [i,i] (along diagonal) are trivially 1s.
    '''
    
    pair2comp = {}
    first_click_num = 5_000
    last_click_num = 0
    click_nums_set = set()
    
    for i in range(preds.shape[0]):
        file1_dir, click1_time, file2_dir, click2_time, label_out, label = preds[i, :]
        click1_num = int(file1_dir.split('/')[-1][10:16])
        click2_num = int(file2_dir.split('/')[-1][10:16])
        
        ### 0 : different whale, 1 : same whale
        ### label_out : prediction, label: actual label (ground truth)
        pair2comp[(click1_num, click2_num)] = label_out
        
        click_nums_set.add(click1_num)
        click_nums_set.add(click2_num) ## need last click num
                
    first_click_num, last_click_num = min(click_nums_set), max(click_nums_set)
    print('first: ', first_click_num, 'last: ', last_click_num, '\n')
        
    check_all_pairs(pair2comp, first_click_num, last_click_num)
    # print('pair2comp dict num of keys: ', len(pair2comp.keys()))
    
    num_clicks = len(click_nums_set)
    # print('number of clicks: ', num_clicks)
    click_nums_sorted = sorted(list(click_nums_set))
    
    idx2click = {idx : click_num for idx, click_num in enumerate(click_nums_sorted)}
    click2idx = {click_num : idx for idx, click_num in enumerate(click_nums_sorted)}
    
    comparisons_matrix = np.zeros((num_clicks, num_clicks), dtype=object)
    for i in range(num_clicks):
        click_i = idx2click[i]
        for j in range(num_clicks):
            click_j = idx2click[j]
            if i == j:
                value = 1 ## comparing a click with itself
            else:
                pair_key = (min(click_i, click_j), max(click_i, click_j)) ## need ordered pair
                value = pair2comp[pair_key]
            comparisons_matrix[i, j] = value
    
    return comparisons_matrix, idx2click, click2idx




def cluster_clicks_iteratively_with_threshold(comparisons_matrix = None, thr = 0.7):
    
    '''
    Iterative clustering with threshold
    Compare with all previous clicks, if % same > threshold, assign click to that cluster
    If multiple clusters pass this threshold, assign click to clueter for which % same was highest.
    '''    
    
    print('threshold: ', thr)
    
    def get_same_percentage_for_existing_clusters(whale2clicks, click_idx):
        whale_to_same_perc = {}
        for whale_num in whale2clicks.keys():
            whale_clicks = sorted(whale2clicks[whale_num])
            score = 0
            for prev_click_idx in whale_clicks:
                if comparisons_matrix[prev_click_idx, click_idx] == 1:
                    score += 1
            same_perc = score / len(whale_clicks)
            whale_to_same_perc[whale_num] = same_perc           
        return whale_to_same_perc
    
    
    num_clicks = comparisons_matrix.shape[0]
    click2whale = {}
    whale2clicks = {}

    click2whale[0] = 1
    whale2clicks[1] = [0,]

    curr_whale_num = 1
    
    for click_idx in range(1, num_clicks):
        new_whale = True
        
        whale_to_same_perc = get_same_percentage_for_existing_clusters(whale2clicks, click_idx)
        print('click idx: ', click_idx, 'similarity percentages: ', whale_to_same_perc)
        
        most_similar_whale_num = 1
        max_same_perc = 0
        for whale_num in whale_to_same_perc.keys():
            if whale_to_same_perc[whale_num] > max_same_perc:
                most_similar_whale_num = whale_num
                max_same_perc = whale_to_same_perc[whale_num]
        
        if max_same_perc >= thr:
            whale_num = most_similar_whale_num
            click2whale[click_idx] = whale_num
            whale2clicks[whale_num].append(click_idx)
            new_whale = False               
        
        if new_whale:
            curr_whale_num = curr_whale_num + 1
            click2whale[click_idx] = curr_whale_num
            whale2clicks[curr_whale_num] = [click_idx,]
                
    return click2whale, whale2clicks
   
    
   
def get_ground_truth_grouping(list_files=None, conv_clicks=None):
    actual_whale2clicks = {}
    actual_click2whale = {}
    # for idx in range(len(list_files)):
    for idx in range(len(conv_clicks)):
        # whale_num = int(list_files[idx][1])
        whale_num = int(conv_clicks[idx][0])        
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
 

def cluster_clicks_using_HCS(conv_clicks, comparisons_matrix, idx2click, det_click_times):
    
    click2whale = {}
    whale2clicks = {}
    
    # G = create_click_graph_from_preds(conv_clicks, comparisons_matrix)
    G = create_click_graph_from_detected_preds(comparisons_matrix, idx2click, det_click_times)
    
    # print(G.nodes.data())
    # print('edges in G: ', len(list(G.edges)))
    
    ## usually only one component
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]    
    # print('number of connected components: ', len(S))
    
    ## doesn't work with many connected components**
    for i in range(len(S)):
        S_i = nx.relabel.convert_node_labels_to_integers(S[i])
        
        # cluster_labels = hcs.labelled_HCS(S_i.copy())
        cluster_labels = hcs.my_improved_labelled_HCS(S_i.copy())
        
        # print('connected comp: ', i, ":")
        # print('cluster labels: , 'cluster_labels)
        # print('num nodes: ', len(cluster_labels))
        print('num clusters: ', len(set(cluster_labels)))
        
        cluster_num_freqs = {}        
        for node_idx in range(len(cluster_labels)):            
            cluster_num = cluster_labels[node_idx]
            
            click2whale[node_idx] = cluster_num
            if cluster_num not in whale2clicks:
                whale2clicks[cluster_num] = [node_idx]
            else:
                whale2clicks[cluster_num].append(node_idx)                
            
            # print(node_idx, ": ", cluster_num)
            # print(idx2click[node_idx], ": ", cluster_num)
            
            if cluster_num not in cluster_num_freqs:
                cluster_num_freqs[cluster_num] = 1
            else:
                cluster_num_freqs[cluster_num] += 1
        
        print('total clicks in each cluster:')
        for cluster_num in sorted(cluster_num_freqs.keys()):
            print('cluster ', cluster_num, ': ', cluster_num_freqs[cluster_num])
        print('---------------------------------------------------------------------------------------\n')
            
    return G, click2whale, whale2clicks



def plot_clustered_clicks_bow_vis(audio_rootname = None, clicks_info_with_det = None, conv_clicks = None, all_codas = None, 
                                  num_clusters = None, num_whales = None, start_time = None, end_time = None, save_dir = None):
    
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
            
            if coda_i[0]["click_time"] != clicks_info_with_det[coda_i[0]["idx"]]["click_time"]:
                print('error: ', coda_i[0]["idx"])            
            first_click_time = coda_i[0]["click_time"]
            
            for j in range(len(coda_i)):
                click_idx = coda_i[j]["idx"]
                if coda_i[j]["click_time"] != clicks_info_with_det[click_idx]["click_time"]:
                    print('error: ', coda_i[j]["idx"])
                
                cluster_num = clicks_info_with_det[click_idx]["det_cluster"]
                
                if cluster_num == None:
                    marker = markers[5]
                else:
                    marker = markers[cluster_num]                   
                
                x = first_click_time
                y = coda_i[j]["click_time"] - first_click_time
                # print('x, y: ', x, y)
                plt.plot([x], [y], marker = marker, color = colour_chart[whale_num], markersize = 8)
        
        
    # custom_lines = [Line2D([1], [1], color='red', ls='-', lw=1.5),
    #                 Line2D([1], [1], color='green', ls='-', lw=1.5)]    
    custom_lines = [Line2D([], [], color=colour_chart[whale_num], ls='-', lw=1.5) 
                     for whale_num in range(1, num_whales + 1)]    
    # list_lab = ['whale 1', 'whale 2']
    list_lab = ['whale ' + str(whale_num) for whale_num in range(1, num_whales + 1)]
    
    custom_lines += [Line2D([], [], color='black', marker=markers[cluster_num], ls='None', markersize=10) 
                     for cluster_num in range(1, num_clusters + 1)]
    list_lab += ['cluster ' + str(cluster_num) for cluster_num in range(1, num_clusters + 1)]
    
    custom_lines += [Line2D([], [], color='black', marker=markers[5], ls='None', markersize=10)]
    list_lab += ['not detected']
    
    plt.legend(custom_lines, list_lab)
    
    plt.xlabel("Time in file")
    plt.ylabel("Time inside coda")
    
    plt.title('file: ' + audio_rootname + ', time: ' + str(start_time) + '-' + str(end_time) + ' sec')
    
    # plt.savefig(save_dir + audio_rootname + '_vis_bow_style_' + str(start_time) + '_' + str(end_time) + '.png')
    plt.savefig(save_dir + audio_rootname + '_' + str(start_time) + '_' + str(end_time) + '.png')        
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
    



def check_file_clicks_against_list_files(file_clicks, list_files, prev_duration = 0):
    # print('file clicks: ', len(file_clicks))
    # print('list files: ', len(list_files))
    
    # print('file clicks is sorted: ', file_clicks == sorted(file_clicks, key=lambda pair : pair[1]))
    # print('list files is sorted on time: ', list_files == sorted(list_files, key=lambda triple : float(triple[2])))
    # print('list files is sorted on ext: ', list_files == sorted(list_files, key = lambda triple: int(triple[0].split('/')[-1][10:16])))
    
    for i in range(len(file_clicks)):
        # (click_audio_dir, whale_num, click_time) = list_files[i]
        if float(list_files[i][2]) + prev_duration != float(file_clicks[i][1]):
            print('issue: ', i, file_clicks[i][1], list_files[i][2]) 
            break

def filter_list_files(list_files, file_clicks, prev_duration = 0):
    file_clicks_times = set(pair[1] for pair in file_clicks)
    print('file clicks all times: ', len(file_clicks_times))    
    new_list_files = []
    for i in range(len(list_files)):
        (click_audio_dir, whale_num, click_time) = list_files[i]        
        if float(click_time) + prev_duration in file_clicks_times:
            new_list_files.append((click_audio_dir, whale_num, click_time))            
    return new_list_files    



def get_matched_clicks(conv_clicks = None, preds = None, epsilon = 100, reuse_preds = False, prev_duration = 0):
    print("--------Getting matched clicks---------------")
    
    print('epsilon: ', epsilon)
    print('reuse preds: ', reuse_preds)
    
    all_positives = set()
    for i in range(preds.shape[0]):
        file1_dir, click1_time, file2_dir, click2_time, label_out, label = preds[i, :]
        if float(click1_time) not in all_positives:
            all_positives.add(float(click1_time))
        if float(click2_time) not in all_positives:
            all_positives.add(float(click2_time))
    all_positives = sorted(list(all_positives))
    
    # print('all positives: ', len(all_positives))
    # print('sorted: ', all_positives == sorted(all_positives))
    # print('first: ',  all_positives[0])
    # print('last: ',  all_positives[-1])
    
    
    # all_labels = [conv_clicks[i][1] for i in range(len(conv_clicks))]
    all_labels = [conv_clicks[i][1] - prev_duration for i in range(len(conv_clicks))]
    
    # print('all labels: ', len(all_labels))
    # print('sorted: ', all_labels == sorted(all_labels))
    # print('first: ',  all_labels[0])
    # print('last: ',  all_labels[-1])
        
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
    
    click_idx_to_det_idx = {}    
    
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
                        #################################
                        click_idx_to_det_idx[i] = j
                        # print('correct mapping: ', abs(22050*conv_clicks[i][1] - 22050*all_positives[j]) <= epsilon)
                        #################################
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
    
    # print('Matched clicks: ', number_of_matches, len(matched_clicks))
    # print('Missed clicks: ', len(missed_clicks))
    # print('total_clicks: ', total_labels)
    # print('match percentage: ', 100 * number_of_matches / len(all_labels_positives))
    # print('duplicate matches frequencies: ', duplicate_match_freqs)
    
    # print('Matched preds: ', len(matched_preds))
    # print('Unmatched preds: ', len(unmatched_preds))
    
    print('--------------------------------------------------')  
    
    
    return missed_clicks, matched_clicks, click_idx_to_det_idx, all_positives



## whale2clicks is only actual_whale2clicks
def group_clicks_to_codas(clicks_info_with_det, conv_clicks, whale2clicks, time_diff_thr):
    all_codas = {}
    for whale_num in whale2clicks.keys():
        click_idxs = whale2clicks[whale_num]
        click_times = []
        for idx in click_idxs:            
            if clicks_info_with_det[idx]["whale_num"] != whale_num:
                print('incosistency: ', idx, clicks_info_with_det[idx]["whale_num"], whale_num)                
            click_time = clicks_info_with_det[idx]["click_time"]
            click_times.append({"idx": idx, "click_time": click_time})
        
        whale_codas = []
        curr_coda = [click_times[0]]
        for i in range(1, len(click_times)):
            if click_times[i]["click_time"] - curr_coda[-1]["click_time"] <= time_diff_thr:
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



def get_total_misclusterings(whale_num_to_cluster_num, actual_click2whale, pred_click2whale, click_idx_to_det_idx):
    total_mis_clusterings = 0
    for click_idx in actual_click2whale.keys():
        whale_num = actual_click2whale[click_idx]
        if click_idx in click_idx_to_det_idx:
            det_idx = click_idx_to_det_idx[click_idx]
            cluster_num = pred_click2whale[det_idx]
            if cluster_num != whale_num_to_cluster_num[whale_num]:
                print('misgrouped: ', click_idx, det_idx, 'whale:', whale_num, 'cluster: ', cluster_num)
                total_mis_clusterings += 1
        else:
            print(click_idx, ' not detected')    
    print('total mis clusterings: ', total_mis_clusterings)




## main method: clusters clicks to speakers, groups clicks into codas, plots clustered clicks in BoW style
def cluster_clicks_to_speakers(audio_rootname, start, end, det_model_version):
    
    def rootname_to_book_num(audio_rootname, books):
        for book_num in range(len(books)):
            book, filename = books[book_num]
            if audio_rootname == filename:
                return book_num
            
    annotations_dir = '/data/scratch/ioannis/click_seperator/list_annotations.p'
    books = pickle.load(open(annotations_dir, 'rb'))    
    key = rootname_to_book_num(audio_rootname, books)
    
    #################################################################################
    
    # audio_rootname = 'sw061b001' ## year 2015
    # audio_rootname = 'sw061b002' ## year 2015    
    # key = 0
    # start = 0
    # end = 235    
    
    print('\n')
    print('---------------------' + audio_rootname + '---------------------\n')
        
    main_dir = '/data/vision/torralba/scratch/ioannis/clustering/'
    
    labels_dir = main_dir + 'click_regress_all_admin/' + audio_rootname + '_labels.p'    
    file_clicks = pickle.load(open(labels_dir, 'rb'))    
    
    # data_ordered_dir = '/data/scratch/ioannis/click_seperator/correct_file_ordered.p'
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
    
    ## list_files is a list of tuples (click_audio_dir, whale_num, click_time) ##
    # list_files = data_ordered[str(key)][start : end]
    list_files = data_ordered[str(key)]    
    list_files = filter_list_files(list_files, file_clicks, prev_duration)
    list_files = list_files[start : end]
    print('filtered list files: ', len(list_files), len(set(list_files)))
    
    conv_clicks = file_clicks[start : end]
    idx2click_gt = {idx : start + idx  for idx in range(len(conv_clicks))}    
    
    actual_click2whale, actual_whale2clicks = get_ground_truth_grouping(list_files, conv_clicks)
    check_dicts(actual_click2whale, actual_whale2clicks)
    
    # check_file_clicks_against_list_files(conv_clicks, list_files, prev_duration)    
    
    
    ##### WORKING WITH DETECTED CLICKS ####
    ## click detector model
    # det_model_version = 'click_reg_wav'
    # det_model_version = 'click_regress_all_noise_3580_17per' ## used for pipeline (test mode)
    
    print('---------------------' + det_model_version + '---------------------\n')
    
    # preds_dir = main_dir + 'detections_click_sep_preds/' + audio_rootname + '_clicks_' + str(start) + '_' + str(end) + '/'
    preds_dir = main_dir + 'detections_click_sep_preds/' + det_model_version + '/' + audio_rootname + '_clicks_' + str(start) + '_' + str(end) + '/'
    i_batch = 0
    preds_file_name = 'batch_' + str(i_batch) + '.p'    
    preds = pickle.load(open(preds_dir + preds_file_name, 'rb'))    
    
    epsilon = 100
    reuse_preds = False
    missed_clicks, matched_clicks, click_idx_to_det_idx, det_click_times = get_matched_clicks(conv_clicks, preds, 
                                                                                              epsilon, reuse_preds,
                                                                                              prev_duration)
    det_idx_to_click_idx = {v : k for k, v in click_idx_to_det_idx.items()}    
    
    print('click idx to det idx number of keys: ', len(click_idx_to_det_idx))
    print('missed clicks: ', missed_clicks)    
    
    #####################################################################################################
    comparisons_matrix, idx2click, click2idx = get_comparisons_matrix(preds)
    clicks_info_with_det = get_clicks_info_with_detections(conv_clicks, idx2click, idx2click_gt, 
                                                           click_idx_to_det_idx, det_click_times)
    #####################################################################################################    
    check_comparisons_matrix(comparisons_matrix)
    print('comparisons matrix: ', comparisons_matrix.shape, '\n') 
    
    ## TODO: implement singletons adoption improvement to HCS algorithm, mentioned in paper
    #####################################################################################################
    G, pred_click2whale, pred_whale2clicks = cluster_clicks_using_HCS(conv_clicks, comparisons_matrix, idx2click, det_click_times)
    ##################################################################################################### 
    check_graph_against_info_with_detections(G, clicks_info_with_det, click_idx_to_det_idx, det_idx_to_click_idx)
    check_dicts(pred_click2whale, pred_whale2clicks)
    
    # print('Predicted grouping: \n')
    # print_grouping(pred_click2whale, pred_whale2clicks, idx2click)    
    
    print('Ground truth grouping: \n')
    print_grouping(actual_click2whale, actual_whale2clicks, idx2click_gt)
    #####################################################################################################    
    
    # whale_num_to_cluster_num = {1: 2, 2: 1, 3: 3}    
    # get_total_misclusterings(whale_num_to_cluster_num, actual_click2whale, pred_click2whale, click_idx_to_det_idx)    
    
    clicks_info_with_det = augment_clicks_info(clicks_info_with_det, click_idx_to_det_idx, pred_click2whale, actual_click2whale)
    
    print('------grouping clicks to codas---------')
    time_diff_thr = 0.7
    all_codas = group_clicks_to_codas(clicks_info_with_det, conv_clicks, actual_whale2clicks, time_diff_thr)    
    print('all codas: ', sum(len(all_codas[whale_num]) for whale_num in all_codas.keys()))
    print('----------------------------------')    
    
    #####################################################################################################
    
    print('--------Plotting codas bow vis style------\n')
    
    plot_save_dir = main_dir + 'codas_bow_vis_style/' + det_model_version + '/'
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)
        
    plot_clustered_clicks_bow_vis(audio_rootname, clicks_info_with_det = clicks_info_with_det, 
                                  conv_clicks = conv_clicks, all_codas = all_codas, 
                                  num_clusters = len(pred_whale2clicks.keys()), 
                                  num_whales = len(actual_whale2clicks.keys()),
                                  save_dir = plot_save_dir)    
    #####################################################################################################




