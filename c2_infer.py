# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 17:32:29 2019

@author: v-hadong
"""

import time
import pandas as pd
import os.path as osp
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, models
from tqdm import tqdm
from utils import give_fake_data


class ModelSpeed(object):
    def __init__(self, model_init, model_predict):
        self.device_opts = core.DeviceOption(caffe2_pb2.CPU, 0) 
        init_def = caffe2_pb2.NetDef()
        path_prefix = './models/c2/'
        model_init = osp.join(path_prefix, model_init)
        model_predict = osp.join(path_prefix, model_predict)
        with open(model_init, 'rb') as f:
            init_def.ParseFromString(f.read())
            init_def.device_option.CopyFrom(self.device_opts)
            workspace.RunNetOnce(init_def.SerializeToString())
        net_def = caffe2_pb2.NetDef()
        with open(model_predict, 'rb') as f:
            net_def.ParseFromString(f.read())
            net_def.device_option.CopyFrom(self.device_opts)
            workspace.CreateNet(net_def.SerializeToString(), overwrite=True)
            
        self.net_name = net_def.name
        
    def test_time(self, data):   
        sum_time = 0
        sum_num = 0
        workspace.FeedBlob("data", data, device_option=self.device_opts)
        for idx in range(100):
            t_start = time.time()
            workspace.RunNet(self.net_name, 1)
            t_end = time.time()
            
            if idx > 0:
                sum_time += t_end - t_start
                sum_num += 1
            if idx == 60:
                break       
        # experiment logs
        bs_time = sum_time / sum_num
        fps = (1 / bs_time) * data.shape[0]
        model_speed_logs.loc[model_speed_logs.shape[0], :] = [model_name, bs, bs_time, fps]


if __name__ == '__main__':
    model_names = [('resnet50_init_net.pb', 'resnet50_predict_net.pb')]
    batch_size = [1, 2, 4, 8]
    model_speed_logs = pd.DataFrame(columns = ['model_name', 'bs', 'bs_time', 'fps'])
        
    #different models
    for model_name in model_names:
        print('-'*15, model_name, '-'*15)
        model_speed = ModelSpeed(model_name[0], model_name[1])
        # different batch size
        for bs in tqdm(batch_size):
            fake_input_data_cl, fake_input_data_cf = give_fake_data(bs)
            model_speed.test_time(fake_input_data_cf)
            
    model_speed_logs.to_csv('./result/c2_model_speed_experiments.csv', index = False)
    
    
    
    