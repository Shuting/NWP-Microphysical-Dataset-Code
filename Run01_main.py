# THIS FILE IS A PART OF NWP MICROPHYSICAL PARAMETERIZATION CODE PROJECT
# Run01_main.py - main code of this project, try to run this code
#
# Created on 2022-04-11
# Copyright (c) Ting Shu
# Email: shuting@gbamwf.com

from Fun01_Dataset import generate_train_dataset, generate_test_dataset
from Fun02_Model import dnn_model
from Fun03_Experiment import train_model, test_model, plot_loss, analyze_results
import os
import tensorflow as tf
import scipy.io as sci
import numpy as np

# experiment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# load training dataset
# use 1 weather processes to train model, you can try a larger number
train_data_set = generate_train_dataset(tr_num=20, each_num=3000)

# build a model
model = dnn_model()

# train model
model_name = 'column_trained_model_0415.h5'
his_name = 'column_his_0415.pickle'
model, his = train_model(train_data_set, model,
                         batch_size=1024, epochs=1000, validation_split=0.2,
                         save_model=True, model_name=model_name, save_history=True, his_name=his_name)

# plot history
plot_name = 'column_train_loss_0415.png'
plot_loss(his_name, save_plot=True, plot_name=plot_name)

# test the trained model
# load test dataset
# you can choose the ID of the test weather process
# test_data_id is in range of [1, 30]
test_data_set = generate_test_dataset(test_data_id=27)

# test model
res_name = 'column_test_res_0415.mat'
pred_label, true_label = test_model(test_data_set, model, do_save=True, res_name=res_name)

# analyze test results
analyze_results(true_label, pred_label, var_id=0, level=1, save_plot=True, plot_name='column_com_res_0415.png')
