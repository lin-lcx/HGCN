import os
import copy
import torch
import joblib
import random
import sys
import time as sys_time
import numpy as np
from lifelines.utils import concordance_index as ci
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_all_ci(pre_time,patient_and_time,patient_sur_type):
    ordered_time, ordered_pred_time, ordered_observed=[],[],[]
    for x in patient_and_time:
        ordered_time.append(patient_and_time[x])
        ordered_pred_time.append(pre_time[x]*-1)
        ordered_observed.append(patient_sur_type[x])
#     print(ci(ordered_time, ordered_pred_time, ordered_observed))
    return ci(ordered_time, ordered_pred_time, ordered_observed)

def get_val_ci(pre_time,patient_and_time,patient_sur_type):
    ordered_time, ordered_pred_time, ordered_observed=[],[],[]
    for x in pre_time:
        ordered_time.append(patient_and_time[x])
        ordered_pred_time.append(pre_time[x]*-1)
        ordered_observed.append(patient_sur_type[x])
#     print(len(ordered_time), len(ordered_pred_time), len(ordered_observed))
    return ci(ordered_time, ordered_pred_time, ordered_observed)
    

def get_patients_information(patients,sur_and_time):
    patient_sur_type = {}
    for x in patients: 
        patient_sur_type[x] = sur_and_time[x][0]
        
    time = []
    patient_and_time = {}
    for x in patients:
        time.append(sur_and_time[x][-1])
        patient_and_time[x] = sur_and_time[x][-1]
        
    kf_label = []
    for x in patients:
        kf_label.append(patient_sur_type[x])
    
    return patient_sur_type, patient_and_time, kf_label
    
    
    
class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(sys_time.strftime('%Y-%m-%d-%H-%M'))
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass    
    
def get_edge_index_full(leng):
    start = []
    end = []
    for i in range(leng):
        for j in range(leng):
            if i!=j:
                start.append(i)
                end.append(j)
    return torch.tensor([start,end],dtype=torch.long).to(device)    
    
def adjust_learning_rate(optimizer, lr, epoch, lr_step=20, lr_gamma=0.5):
    lr = lr * (lr_gamma ** (epoch // lr_step)) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    