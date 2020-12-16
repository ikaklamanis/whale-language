import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import io, os
from torch.utils.data import Dataset, DataLoader
import pickle
from IPython import embed
from tensorboardX import SummaryWriter
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

from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
# from plot_confusion_matrix import make_confusion_matrix




######################## Helper functions ######################    

class sample_data(Dataset):
    def __init__(self, data_in,data_ord): 
        self.data_in = data_in
        self.data_ord = data_ord

    def __len__(self):
        return len(self.data_in)
    
    def __getitem__(self, idx):
        
        ## only for test mode
        
        audio_dir_1, label_1 = self.data_in[idx, 0], self.data_in[idx, 2]
        audio_dir_2, label_2 = self.data_in[idx, 4], self.data_in[idx, 6]
        time_1 = float(self.data_in[idx, 3])
        time_2 = float(self.data_in[idx, 7])        
        
        audio1, sr = librosa.load(audio_dir_1, mono=False)            
        # find time of click's peak?
        start_1 = 10925 + np.argmax(abs(audio1[1 , 10925 : 11035])) # why dim 1 and not 0?
                        
        audio2, sr = librosa.load(audio_dir_2, mono=False)
        start_2 = 10925 + np.argmax(abs(audio2[1 , 10925 : 11035]))           
        
        audio = np.concatenate((audio1[:, start_2 : start_2 + 300], audio2[:, start_1 : start_1 +300]), axis=1)

        if int(label_1) == int(label_2):
            label = 1
        else:
            label = 0
            
        ## return audio, label, click_1_file_dir, click_1_time, click_2_file_dir, click_2_time 
        return (audio, label, audio_dir_1, time_1, audio_dir_2, time_2)
        
    
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
        self.linear = nn.Linear(512, 2)
    
    def forward(self, input_audio):
        output = self.linear(input_audio)
        return output



############################### Main method: click separator in test mode ######################



def run_click_separator_test_mode(audio_rootname, sep_model_version, sep_model_load_dir, exp_name, det_model_version,
                                  start, end):
    
    
    
    ############ Admin work (directories) ###################################################   
    
    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')
    if not os.path.exists(os.path.join('./ckpts', exp_name)):
        os.makedirs(os.path.join('./ckpts',exp_name))

    ###### Dataset Loading and Splitting##########
    
    data_directory = '/data/vision/torralba/scratch/ioannis/clustering/click_separator_training/correct_data_same_click_diff_click_correct_times.p'
    total_data = pickle.load(open(data_directory,"rb"))
    
    data_ordered_dir = '/data/vision/torralba/scratch/ioannis/clustering/click_separator_training/file_ordered_correct_times.p'
    file_ordered = pickle.load(open(data_directory,"rb"))    
    
    #######################################################################################################    
    # audio_rootname = 'sw061b001'    
    # start = 0
    # end = 235    
    
    print('------Running click separator on detected clicks------\n')    
    print('Clicks: ', start, '-', end-1, '\n')    
    
    main_dir = '/data/vision/torralba/scratch/ioannis/clustering/'    
    
    # test_pick = main_dir + 'custom_test_pick_preds/' + audio_rootname + '/' + audio_rootname + '_clicks_' + str(start) + '_' + str(end) + '.p'
    test_pick = main_dir + 'custom_test_pick_preds/' + det_model_version + '/' + audio_rootname + '/' + audio_rootname + '_clicks_' + str(start) + '_' + str(end) + '.p'
    audio_recordings_test = pickle.load(open(test_pick,"rb"))    
    
    # preds_save_dir = main_dir + 'detections_click_sep_preds/' + audio_rootname + '_clicks_' + str(start) + '_' + str(end) + '/'
    preds_save_dir = main_dir + 'detections_click_sep_preds/' + det_model_version + '/' + audio_rootname + '_clicks_' + str(start) + '_' + str(end) + '/'
    
    if not os.path.exists(preds_save_dir):
        os.makedirs(preds_save_dir)
    
    ############ End of admin work (directories) ###################################################
        
    
    np.random.seed(0)
    torch.manual_seed(0)
    
    seq = SoundNet()
    # seq = clickdetector()
    seq.cuda()
    # seq = nn.DataParallel(seq)
    valnet = value_net()
    valnet.cuda()
    # valnet = nn.DataParallel(valnet)
    
    # optimizer2 = optim.Adam(valnet.parameters(), lr=args.lr, weight_decay=args.weightdecay)   
    # optimizer = optim.Adam(seq.parameters(), lr=args.lr, weight_decay=args.weightdecay)    
    # criterion = nn.CrossEntropyLoss()    
    
    test_dataset = sample_data(audio_recordings_test, file_ordered)    
    print('test dataset length: ', len(test_dataset))
    
    test_dataloader = DataLoader(test_dataset, batch_size = len(test_dataset),
                            shuffle = False, num_workers = 20)

    # predictions = []
      
    checkpoint = torch.load(sep_model_load_dir) # NEED TO CHANGE
    
    seq.load_state_dict(checkpoint['state_dict'])
    valnet.load_state_dict(checkpoint['state_dict_valnet'])
    seq.eval()
    valnet.eval()
    
    for i_batch, sample_batched in enumerate(test_dataloader): ### NEEDS CHANGEEEEEEEEE
    
        print(i_batch)
        
        # optimizer.zero_grad()
        # optimizer2.zero_grad()
        audio = sample_batched[0].type(torch.cuda.FloatTensor) 
        label = sample_batched[1].type(torch.cuda.FloatTensor)
        
        click_1_file_dir, click_1_time, click_2_file_dir, click_2_time = sample_batched[2:] ## NEW
        
        out = valnet(seq(audio))
        
        ## NEW ##
        
        out = out.cpu().data.numpy()
        labels_out = np.argmax(out,axis = 1)
        label = label.cpu().data.numpy()
        
        preds = np.array([list(click_1_file_dir), list(click_1_time),
                          list(click_2_file_dir), list(click_2_time),
                          labels_out, label], dtype=object)
        preds = preds.T
        print('predictions np array shape: ', preds.shape)
        
        preds_dir = preds_save_dir
        pickle.dump(preds, open(preds_dir + 'batch_' + str(i_batch) + '.p', "wb"))
        
        cf_matrix_test = confusion_matrix(label, labels_out)
        
        acc = 0
        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(labels_out.shape[0]):
            if labels_out[i] == label[i]:
                acc += 1
                
            if labels_out[i] == 1 and label[i] == 1:
                tp += 1
            if labels_out[i] == 0 and label[i] == 0:
                tn += 1
            if labels_out[i] == 1 and label[i] == 0:
                fp += 1
            if labels_out[i] == 0 and label[i] == 1:
                fn += 1            
        
        print('accuracy: ', acc / labels_out.shape[0])
        print("Number of pairs same whale: ", np.sum(label))
        print("Percentage of same whale: ", np.sum(label) / len(label) * 100)
        
        print('TP: ', tp)
        print('TN: ', tn)
        print('FP: ', fp)
        print('FN: ', fn)
        
        print ('Confusion Matrix :')
        print(cf_matrix_test)
