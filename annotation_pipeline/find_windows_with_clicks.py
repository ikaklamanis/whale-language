# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 23:47:49 2020

@author: jkakl
"""

import numpy as np
import io, os
import pickle
import argparse
import math
import shutil
import seaborn as sn
import pandas as pd
import librosa
import cv2
import random
from collections import Counter


def group_click_times_to_windows(click_time_preds, distance = 60):
    windows = []
    curr_window = [click_time_preds[0]]
    for i in range(1, len(click_time_preds)):
        pred_time = click_time_preds[i]


def find_windows_with_clicks(audio_rootname, click_time_preds, distance = 3_000, ):
    windows = []
    
    for i in range(len(click_time_preds)):
        pred_time = click_time_preds[i]
        
    
    
    return windows