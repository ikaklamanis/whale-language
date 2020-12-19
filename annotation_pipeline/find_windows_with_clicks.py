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


## TODO: code to filter audio file and extract time windows with clicks based on detections of 1st click detector


def group_click_times_to_windows(click_time_preds, distance = 60):
    raise NotImplementedError


def find_windows_with_clicks(audio_rootname, click_time_preds, distance = 3_000):
    raise NotImplementedError