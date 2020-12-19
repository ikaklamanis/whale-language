import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import io, os
from torch.utils.data import Dataset, DataLoader
import pickle
from IPython import embed
import argparse
import random
import torch
from torch.autograd import Variable
import h5py
from torchvision import datasets, models, transforms
import math
import shutil
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import librosa
import librosa.display
import cv2
import random



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



######################## Helper functions ######################

## only for test mode
class sample_data(Dataset):
    def __init__(self, data_in): 
        self.data_in = data_in

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, idx):
        
        curr_audio, sr = librosa.load(self.data_in[idx,0][1], mono=False) ## middle file path
        next_audio = np.zeros((curr_audio.shape[0], window_size), dtype=np.float32) ## shape = (2,2000)
        
        _next = True if self.data_in[idx,0][2] != None else False 
        if _next:
            next_audio, sr = librosa.load(self.data_in[idx,0][2], mono=False)
            
        cut = window_size ## green window size
            
        audio = np.concatenate((curr_audio, next_audio[:, :cut]), axis=1) # (2,24050)
            
        label_keys = self.data_in[idx,1]
            
        return (audio, label_keys)
    
            
###### Model #################################

class SoundNet(nn.Module):
    def __init__(self):
        super(SoundNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1),
                               padding=(32, 0))
        self.batchnorm1 = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1),
                               padding=(16, 0))
        self.batchnorm2 = nn.BatchNorm2d(32, eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((8, 1), stride=(8, 1))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1),
                               padding=(8, 0))
        self.batchnorm3 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1),
                               padding=(4, 0))
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-5, momentum=0.1)
        self.relu4 = nn.ReLU(True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm5 = nn.BatchNorm2d(256, eps=1e-5, momentum=0.1)
        self.relu5 = nn.ReLU(True)
        self.maxpool5 = nn.MaxPool2d((4, 1), stride=(4, 1))

        self.conv6 = nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm6 = nn.BatchNorm2d(512, eps=1e-5, momentum=0.1)
        self.relu6 = nn.ReLU(True)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1),
                               padding=(2, 0))
        self.batchnorm7 = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1)
        self.relu7 = nn.ReLU(True)

        self.conv8_objs = nn.Conv2d(1024, 1000, kernel_size=(8, 1),
                                    stride=(2, 1))
        self.conv8_scns = nn.Conv2d(1024, 401, kernel_size=(8, 1),
                                    stride=(2, 1))

    def forward(self, waveform):
        x = self.conv1(waveform.unsqueeze(1).permute(0,1,3,2))
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        x = x.reshape(x.shape[0],-1)
        return x
    
    
class value_net(nn.Module):
    def __init__(self, symmetric=True):
        super(value_net, self).__init__()
        self.linear = nn.Linear(768, black_window_size + 2)
    
    def forward(self, input_audio):
        output = self.linear(input_audio)
        return output

################################ Main method: click detector in test mode #########################################


def run_click_detector_test_mode(audio_rootname, model_version, model_load_dir, exp_name):
    '''
    Run click detector model (in test mode) to parse the entire audio file 'audio_rootname'
    - model_version: click detector version name, to be used in naming directory to save predictions
    - model_load_dir: directory from which to load trained click detector model version
    - exp_name: experiment name, not important.
    
    Effect: saves detections in batches (usually only 1 batch) in pickle files in the following directory:
        
        '/data/vision/torralba/scratch/ioannis/click_regress/training/detector_noise_right_edges_annot_data/' 
         + model_version + '/' + audio_rootname + '/' + 'batch_' + str(i_batch) + '.p'
    '''
    
    ############ Admin work (directories) ###################################################
    
    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')    
    if not os.path.exists(os.path.join('./ckpts', exp_name)):
        os.makedirs(os.path.join('./ckpts', exp_name))    
    
    ###### Dataset Loading and Splitting##########
    
    data_directory = '/data/scratch/ioannis/new_annotation.p'
    total_data = pickle.load(open(data_directory,"rb"))    
    
    main_dir = '/data/vision/torralba/scratch/ioannis/clustering/'
    
    test_pick = main_dir + 'click_regress_all_admin/' + 'annot_audio_' + audio_rootname + '.p' ## also triples
    audio_recordings_test = pickle.load(open(test_pick,"rb")) ## MOST RECENT !!!! - for testing    
    
    new_detector_dir = '/data/vision/torralba/scratch/ioannis/click_regress/training/detector_noise_right_edges_annot_data/' 
    save_dir = new_detector_dir + model_version + '/' + audio_rootname + '/'
    
    print('model version: ', model_version)
    print('dir: ', model_load_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)        
        
    ############ End of admin work (directories) ###################################################
    
    np.random.seed(0)
    torch.manual_seed(0)
    
    seq = SoundNet()
    valnet = value_net()
    valnet.cuda()
    valnet = nn.DataParallel(valnet)
    # optimizer2 = optim.Adam(valnet.parameters(), lr=args.lr, weight_decay=args.weightdecay)  
    
    seq.cuda()
    seq = nn.DataParallel(seq)  
    
    ## assumes using CrossEntropyLoss ##
    
    # optimizer = optim.Adam(seq.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    # criterion = nn.CrossEntropyLoss()    
    
    test_dataset = sample_data(audio_recordings_test)
    print('test dataset length: ', len(test_dataset))    
    test_dataloader = DataLoader(test_dataset, batch_size = len(test_dataset),
                        shuffle = False, num_workers = 20)    
    # print('GOOD 1')
    
    ############################################################################################
    
    print('Running in test mode, file: ', audio_rootname)
    # checkpoint = torch.load('/data/scratch/ioannis/ckpts/click_reg_wav/499.pth.tar') ## this is trained w/o noise examples
    checkpoint = torch.load(model_load_dir)
    
    valnet.load_state_dict(checkpoint['state_dict_valnet'])
    valnet.eval()
    seq.load_state_dict(checkpoint['state_dict'])
    seq.eval()   
    
    print('loaded model parameters')    
    
    predictions = []    
    
    for i_batch, sample_batched in enumerate(test_dataloader):
            
        audio = sample_batched[0].type(torch.cuda.FloatTensor)
        # print('audio: ', audio.shape)        
        batch_size = audio.shape[0] 
        # print('batch size: ', batch_size)        
        n_crops = (audio.shape[2] - window_size) - min_timestep
        preds_np = np.zeros((batch_size, n_crops))

        label = sample_batched[1]
        preds = []        
        # print('audio: ', audio.shape) ## attention: audio.shape = (batch_size, 2, 22050)
        for i in range(min_timestep, audio.shape[2] - window_size):
            crop_audio = audio[: , : , i : i + window_size]          
            if i % 2000 == 0:
                print(i)                
            out = seq(crop_audio)
            out = valnet(out)
            out = out.cpu().data.numpy()
            
            labels_out = np.argmax(out,axis = 1)
            # print('labels out: ', labels_out.shape) ## shape = (24,)
            preds.append(labels_out)
            preds_np[:, i] = labels_out                
        # print('preds_np: ', preds_np.shape)
        predictions.append(preds)
        
        preds_save_dir = save_dir + 'batch_' + str(i_batch) + '.p'
        # print('preds_save_dir: ', preds_save_dir)
        
        pickle.dump(preds_np, open(preds_save_dir, "wb"))
        
        print('----------Batch ' + str(i_batch) + ': done--------------')
        
        
        
        
        
        
        
