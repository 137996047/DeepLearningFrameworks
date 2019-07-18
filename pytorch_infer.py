#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 12:28:30 2018

@author: hailin
"""
import time
import torch
import pandas as pd
import torchvision.models as models
from utils import give_fake_data
from tqdm import tqdm


class ModelSpeed(object):
    def __init__(self, model):
            self.cuda_is_available = torch.cuda.is_available()
            self.device = torch.device('cpu')
            self.model = model.to(self.device)

    def test_time(self, data):   
        # generate inputs data
        inputs = torch.tensor(data).to(self.device)
        # use mkldnn accelerator
        #inputs = inputs.to_mkldnn()
        self.model.eval()
        with torch.no_grad():
            sum_time = 0
            sum_num = 0
            for idx in range(50):
                #keep t_start, model_inference, t_end procedure synchronize
                if self.cuda_is_available:
                    torch.cuda.synchronize()
                t_start = time.time()
                self.model(inputs)
                if self.cuda_is_available:
                    torch.cuda.synchronize()
                t_end = time.time()
                
                if idx > 0:
                    sum_time += t_end - t_start
                    sum_num += 1
                if idx == 40:
                    break       
            # experiment logs
            bs_time = sum_time / sum_num
            fps = (1 / bs_time) * data.shape[0]
            model_speed_logs.loc[model_speed_logs.shape[0], :] = [model_name, bs, bs_time, fps]


if __name__ == '__main__':
    model_names = ['resnet18', 'resnet50']
    batch_size = [1, 2, 4, 8]
    model_speed_logs = pd.DataFrame(columns = ['model_name', 'bs', 'bs_time', 'fps'])
    # set dtype include input_data and model_parameters
    torch.set_default_dtype(torch.float)
    
    #different models
    for model_name in model_names:
        print('-'*15, model_name, '-'*15)
        model = getattr(models, model_name)(pretrained=True)
        model_speed = ModelSpeed(model)
        
        # different batch size
        for bs in tqdm(batch_size):
            data_cl,data_cf = give_fake_data(bs)
            model_speed.test_time(data_cf)
            
    model_speed_logs.to_csv('./result/pytorch_model_speed_experiments.csv', index = False)
    
    
    
    