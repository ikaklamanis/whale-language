# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
import math
import librosa
import librosa.display
from scipy.io import wavfile
import io, os



# Reminder of the data structure:
#   book[j,0]  Who made the sound (ID number)
#   book[j,1]  What time did it start at
#   book[j,2:] Codas



annotations_dir = '/data/scratch/ioannis/click_seperator/list_annotations.p'
books = pickle.load(open(annotations_dir, 'rb'))

directories_dir = '/data/scratch/ioannis/click_seperator/directories_dict.p'
directories =  pickle.load(open(directories_dir, 'rb'))


def get_audio_durations():
    durations_dict = {}
    file_num = 0
    for (book, filename) in books:        
        print('Reading filename ' + str(file_num) + ': ' + filename)
        print('-----------------------------------------')
        # read the file and get the sample rate and data
        file_dir = directories[filename]
        # rate, data = wavfile.read(file_dir)
        try:
            data, rate = librosa.load(file_dir, mono=False)
            duration = data.shape[1] / rate ## total file duration (seconds) ## CHANGE THIS        
            durations_dict[filename] = {'duration': duration, 'rate': rate}
            print('duration: ', duration)
        except:
            print('Error loading filename ', filename)
        
        file_num += 1
        
        audio_durations_dir = '/data/vision/torralba/scratch/ioannis/clustering/click_regress_all_admin/audio_files_durations.p'
        pickle.dump(durations_dict, open(audio_durations_dir, 'wb'))
    
    return durations_dict 


audio_durations = get_audio_durations()

print(audio_durations, len(audio_durations))

audio_durations_dir = '/data/vision/torralba/scratch/ioannis/clustering/click_regress_all_admin/audio_files_durations.p'
pickle.dump(audio_durations, open(audio_durations_dir, 'wb'))


