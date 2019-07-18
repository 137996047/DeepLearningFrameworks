# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:36:13 2019

@author: v-hadong
"""

import sys
import numpy as np
import torch
import keras as K
import tensorflow as tf
import caffe2 as c2
import onnxruntime as ort
from multiprocessing import cpu_count

ITER_NUMS = 25

def give_fake_data(batch_size):
    """ Create an array of fake data to run inference on"""
    
    np.random.seed(10)
    data = np.random.rand(batch_size, 224, 224, 3).astype(np.float32)
    return data, np.swapaxes(data, 1, 3)

def yield_mb_X(X, batchsize):
    """ Function yield (complete) mini_batches of data"""
    for i in range(len(X)//batchsize):
        yield i, X[i*batchsize:(i+1)*batchsize]
        
def get_system_information():
        print("Python: ", sys.version)
        print("Numpy: ", np.__version__)
        # NumberOfLogicalProcessors 
        # print("cpu count: ", cpu_count())

def deep_framwork_version():
    print('pytorch version:', torch.__version__)
    print('caffe2 version:', torch.__version__)
    print('tensorflow version:', tf.__version__)
    print('keras version:', K.__version__)
    print('onnxruntime version:', ort.__version__)
    
if __name__ == '__main__':
    get_system_information()
    deep_framwork_version()