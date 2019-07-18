# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 15:45:53 2019

@author: WaterWood
"""

import time
import tensorflow as tf
import os.path as osp
import pandas as pd
from tqdm import tqdm
from utils import give_fake_data, ITER_NUMS
from keras.applications import resnet50 as models

class ModelSpeed(object):
    def __init__(self, model_name):
        self.model = getattr(models, model_name)(include_top=True, weights='imagenet', input_shape=(224,224,3))
        
    def test_time(self, data):   
        sum_time = 0
        sum_num = 0
        
        for idx in range(ITER_NUMS):
            t_start = time.time()
            self.model.predict_on_batch(data)
            t_end = time.time()
            if idx >= 5:
                sum_time += t_end - t_start
                sum_num += 1    

        # experiment logs
        bs_time = sum_time / sum_num
        fps = (1 / bs_time)*data.shape[0]
        model_speed_logs.loc[model_speed_logs.shape[0], :] = [model_name, bs, bs_time, fps]


if __name__ == '__main__':
    model_names = ['ResNet50']
    batch_size = [1, 2, 4, 8]
    model_speed_logs = pd.DataFrame(columns = ['model_name', 'bs', 'bs_time', 'fps'])
    
    #different models
    for model_name in model_names:
        print('-'*15, model_name, '-'*15)
        model_speed = ModelSpeed(model_name)
        time.sleep(1)
        # different batch size
        for bs in tqdm(batch_size):
            fake_input_data_cl, fake_input_data_cf = give_fake_data(bs)
            model_speed.test_time(fake_input_data_cl)
            
    model_speed_logs.to_csv('./result/keras_model_speed_experiments.csv', index = False)