# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:09:26 2022

@author: andre
"""
import numpy as np
import tensorflow as tf

#tf.enable_eager_execution(tf.ConfigProto(log_device_placement=True)) 

#print(tf.add([1.0, 2.0], [3.0, 4.0])) 

print(tf.test.is_gpu_available())
