# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:06:49 2019

@author: v-hadong
"""

import time
import tensorflow as tf
import os.path as osp
import pandas as pd
from tqdm import tqdm
from utils import give_fake_data
from tensorflow.contrib.slim.nets import resnet_v2


class ModelSpeed(object):
    def __init__(self, model_name):
        self.input_tensor = tf.placeholder(tf.float32, shape=(None,224,224,3), name='input_image')
        self.sess = tf.Session()
        arg_scope = resnet_v2.resnet_arg_scope()
        with tf.contrib.slim.arg_scope(arg_scope):
            self.net, end_points = getattr(resnet_v2, model_name)(self.input_tensor, 1001, is_training=False)
        saver = tf.train.Saver()
        saver.restore(self.sess, osp.join('./models/tf/', model_name + '.ckpt'))
        
    def test_time(self, data):   
        sum_time = 0
        sum_num = 0
        for idx in range(60):
            t_start = time.time()
            self.sess.run(self.net, feed_dict={self.input_tensor: data})
            t_end = time.time()
            
            if idx > 10:
                sum_time += t_end - t_start
                sum_num += 1    
            
        # experiment logs
        bs_time = sum_time / sum_num
        fps = (1 / bs_time) * data.shape[0]
        model_speed_logs.loc[model_speed_logs.shape[0], :] = [model_name, bs, bs_time, fps]


if __name__ == '__main__':
    model_names = ['resnet_v2_50']
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
            
    model_speed_logs.to_csv('./result/tf_model_speed_experiments.csv', index = False)