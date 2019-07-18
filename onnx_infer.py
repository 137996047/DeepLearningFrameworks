# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 09:07:18 2019

@author: v-hadong
"""

import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import time
import pandas as pd
from tqdm import tqdm
from utils import give_fake_data


class ModelSpeed(object):
    def __init__(self, model_name):
        self.session = onnxruntime.InferenceSession('D:/frameworks/models/onnx/' + model_name + '.onnx', None)
        self.input_name = self.session.get_inputs()[0].name  
        self.output_name = self.session.get_outputs()[0].name
        print('input_name:', self.input_name)
        print('output_name:', self.output_name)
        print('output_shape:', self.session.get_outputs()[0].shape)
        
    def test_time(self, data):   
        sum_time = 0
        sum_num = 0
        for idx in range(100):
            t_start = time.time()
            self.session.run([self.output_name], {self.input_name: data})
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
    model_names = ['resnet18v2', 'resnet50v2']
    batch_size = [1, 2, 4, 8]
    model_speed_logs = pd.DataFrame(columns = ['model_name', 'bs', 'bs_time', 'fps'])
        
    #different models
    for model_name in model_names:
        print('-'*15, model_name, '-'*15)
        model_speed = ModelSpeed(model_name)
        # different batch size
        for bs in tqdm(batch_size):
            fake_input_data_cl, fake_input_data_cf = give_fake_data(bs)
            model_speed.test_time(fake_input_data_cf)
            
    model_speed_logs.to_csv('D:/frameworks/result/onnx_model_speed_experiments.csv', index = False)
    
    
    
    