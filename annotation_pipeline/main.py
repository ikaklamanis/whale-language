# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 23:36:28 2020

@author: jkakl
"""

from crop_annot_audio import *
from get_click_times_labels import *
from click_regress_test_mode import *
from relocate_preds import *
from compare_model_detections import *
from make_detected_click_crops import *
from custom_test_pick import *
from click_differentiator_test_mode import *
from cluster_detected_clicks_to_speakers import *


def pipeline(audio_rootname, det_model_version, det_model_load_dir, sep_model_version, sep_model_load_dir,
             first_click_num, last_click_num):
    '''
    Pipeline method for detecting clicks, clustering them into speakers, and plotting clustered clicks.
    Makes use of several methods imported from the python files above.
    The pipeline can be broken down to 10 main steps. Each step is commented in the code.
    Currently the pipeline is set up in a way that each step can be executed only if the previous steps have been executed.
    

    Parameters
    ----------
    audio_rootname : string
        DESCRIPTION. audio filename.
    det_model_version : string
        DESCRIPTION. click detector model version name.
    det_model_load_dir : string
        DESCRIPTION. directory from which to load trained click detector model version.
    sep_model_version : string
        DESCRIPTION. click separator model version name.
    sep_model_load_dir : string
        DESCRIPTION. directory from which to load trained click separator model version.
    first_click_num : int
        DESCRIPTION. number of first click in range [first click num, last click num].
    last_click_num : int
        DESCRIPTION. number of first click in range [first click num, last click num].

    Returns
    -------
    None.

    '''
    
    main_dir = '/data/vision/torralba/scratch/ioannis/clustering/'
    
    
    ## 1) Crop audio file to 1-sec files
    print('\n---------- 1) Cropping audio rootname: ', audio_rootname, ' ---------------- \n')
    
    dataset_directory = '/data/scratch/ioannis/dataset/'
    cropped_audio_save_dir = '/data/vision/torralba/scratch/ioannis/dataset/cropped_files_with_annot/'
    
    crop_audio(audio_rootname, dataset_directory, cropped_audio_save_dir)
    ## annot audio    
    file_paths = create_annot_audio_dirs_augmented(audio_rootname, cropped_audio_save_dir, sort = True, augment = True)
    
    file_paths = np.array(file_paths, dtype=object) ## this is correct
    print('annot dirs shape: ', file_paths.shape)    
    annot_dir = main_dir + 'click_regress_all_admin/' + 'annot_audio_' + audio_rootname + '.p'
    pickle.dump(file_paths, open(annot_dir, 'wb'))
    
    
    ## 2) Get ground truth labels for audio file (local and global ?)
    print('\n---------- 2) Saving ground truth click times/labels ---------------- \n')
    
    ######################################################################################################################################
    save_ground_truth_click_times(audio_rootname) ## file clicks
    save_global_and_local_labels(audio_rootname) ## global and local annot audio    
    ######################################################################################################################################
    
    
    ## 3) Run click detector model on entire audio file (on all cropped audio files)
    print('\n---------- 3) Running click detector: ', det_model_version, ' ---------------- \n')
    
    detector_exp_name = 'test_' + audio_rootname + '_pipeline'
    ######################################################################################################################################
    run_click_detector_test_mode(audio_rootname, det_model_version, det_model_load_dir, detector_exp_name)
    ######################################################################################################################################
    
    
    ## 4) Relocate detections
    
    print('\n---------- 4) Relocating predictions ------------\n')
    print('audio rootname: ', audio_rootname, '-------------------')
    
    audio_annot_directory = main_dir + 'click_regress_all_admin/' + 'annot_audio_' + audio_rootname + '.p' ## also triples
    # audio_annot_dirs = pickle.load(open(audio_annot_directory,"rb"))
    
    new_detector_dir = '/data/vision/torralba/scratch/ioannis/click_regress/training/detector_noise_right_edges_annot_data/' 
    det_save_dir = new_detector_dir + det_model_version + '/' + audio_rootname + '/'    
    reloc_dir = new_detector_dir + 'annot_right_edges_reloc/' + det_model_version + '/'
    ######################################################################################################################################
    relocate_predictions(audio_rootname, audio_annot_directory, det_save_dir, reloc_dir)
    ######################################################################################################################################
    
        
    ## 5) Parse, average, and convert detections to global times (methods from 'click_detector_entire_visual')
    print('\n---------- 5) Parsing/averaging detections ---------------- \n')
    
    conf_number = 15
    ######################################################################################################################################
    all_positives = click_detector_parse_entire_file(audio_rootname = audio_rootname, model_preds_dir = reloc_dir, 
                                                      conf_number = conf_number, model_version = det_model_version)
    ######################################################################################################################################
    
    
    ## 6) TODO: extract windows with clicks, and repeat steps 3-4-5 on those windows 
    ##            (with more sensitive click detector model)
    
    
    
    ## 7) Make crops of detected clicks for click separator
    print('\n---------- 7) Making audio crops for click separator ---------------- \n')        
 
    noise_visual_dir = new_detector_dir + 'models_trained_with_noise_detections_visuals/' + det_model_version + '/'
    # all_positives_save_dir = '/data/vision/torralba/scratch/ioannis/clustering/click_regress_all_admin/' + audio_rootname + '_all_positives_conf_' + str(conf_number) + '.p'
    all_positives_save_dir = noise_visual_dir + audio_rootname + '_all_positives_conf_' + str(conf_number) + '.p'
    global_click_time_preds = pickle.load(open(all_positives_save_dir, 'rb'))
    
    cropped_det_clicks_save_dir = '/data/vision/torralba/scratch/ioannis/clustering/cropped_detected_clicks/' + det_model_version + '/'
    
    ######################################################################################################################################
    detected_clicks_annot = crop_audio_clicks(file_to_crop = audio_rootname, click_time_preds = global_click_time_preds, 
                                              save_dir = cropped_det_clicks_save_dir)
    ######################################################################################################################################
    print('detected clicks annot data: ', detected_clicks_annot.shape)    
    # detected_clicks_annot_dir = '/data/vision/torralba/scratch/ioannis/clustering/click_regress_all_admin/detected_clicks_annot_data_' + audio_rootname + '.p'
    detected_clicks_annot_dir = noise_visual_dir + 'detected_clicks_annot_data_' + audio_rootname + '.p'    
    pickle.dump(detected_clicks_annot, open(detected_clicks_annot_dir, 'wb'))    
    ######################################################################################################################################
    
    
    ######### Specify a time window in the audio file: either [click #, click #] or [start_time, end_time]
        
    start = first_click_num
    end = last_click_num + 1
    
    ## 8) Generate directories (test pick) for click separator (on specified window)
    print('\n---------- 8) Generating directories for click separator ---------------- \n')
    
    custom_preds_data_save_dir = main_dir + 'custom_test_pick_preds/' + det_model_version + '/' + audio_rootname + '/'    
    ######################################################################################################################################   
    det_annot_data_in_range = get_det_annot_data_in_range(audio_rootname, det_model_version, start, end)
    
    custom_data_in = create_custom_data_in_from_detections(audio_rootname, det_annot_data_in_range, 
                                                            first_click_num, last_click_num + 1, 
                                                            custom_preds_data_save_dir)
    ######################################################################################################################################
    
    
    
    ## 9) Run click separator model on audio crops
    print('\n---------- 9) Running click separator: ', sep_model_version, ' ---------------- \n')
    
    separator_exp_name = 'test_' + audio_rootname + '_pipeline_' + str(start) + '_' + str(end)
    
    ######################################################################################################################################
    run_click_separator_test_mode(audio_rootname, sep_model_version, sep_model_load_dir, separator_exp_name, det_model_version,
                                  start, end)
    ######################################################################################################################################
    
    
    ## 10) Cluster clicks to speakers, group into codas, and plot clustered clicks bow-vis-style
    print('\n---------- 10) Clustering clicks to speakers and plotting ---------------- \n')
    
    cluster_clicks_to_speakers(audio_rootname, start, end, det_model_version)
    
    
    ## 10b) Cluster and plot only based on detections/predictions (will be used for un-annotated data)





if __name__ == '__main__':
    
    ## specify audio rootname and range of click numbers (first-last click num)
    
    # audio_rootname = 'sw061b002'
    audio_rootname = 'sw061b003'
    # audio_rootname = 'sw090b002'
    
    first_click_num = 210
    last_click_num = 420
    
    ## specify model versions for click detector and click separator    
    
    # det_model_version = 'click_regress_all_noise_3580_17per' ## trained with hard (noise) examples
    # det_model_load_dir = '/data/vision/torralba/scratch/ioannis/click_regress/training/ckpts/click_regress_all_noise_3580_17per/499.pth.tar'
    
    det_model_version = 'click_reg_wav' ## not trained with hard (noise) examples
    det_model_load_dir = '/data/scratch/ioannis/ckpts/click_reg_wav/499.pth.tar'
    sep_model_version = 'correct_cnn_cfs'
    sep_model_load_dir = '/data/scratch/ioannis/click_seperator/CNN_separators/ckpts/correct_cnn_cfs/499.pth.tar'
    # save_dir = None
    
    print('\n', 'RUNNING PIPELINE on audio file: ', audio_rootname, '\n')
    
    ## main method (pipeline), can be broken down to 10 steps
    pipeline(audio_rootname, det_model_version, det_model_load_dir, sep_model_version, sep_model_load_dir,
             first_click_num, last_click_num)


