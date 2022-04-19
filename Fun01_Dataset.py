# THIS FILE IS A PART OF NWP MICROPHYSICAL PARAMETERIZATION CODE PROJECT
# Fun01_Dataset.py - generate dataset from mat file, pre-process and normalize data, generate training and test datasets
#
# Created on 2022-04-11
# Copyright (c) Ting Shu
# Email: shuting@gbamwf.com

import numpy as np
import scipy.io as sci
import os


# generate training datasets
def generate_train_dataset(tr_num=20, each_num=0):
    dirs = os.listdir('./all_mp_mat/')
    dirs.sort(key=list_sort)
    file_name_list = dirs[:tr_num]
    all_input = np.zeros((1, 50, 15))
    all_output = np.zeros((1, 50, 12))
    for file_name in file_name_list:
        cur_mat_file = sci.loadmat('./all_mp_mat/'+file_name)
        # model input, n*50*15
        cur_x = np.reshape(cur_mat_file['x'], [478*574, 50, 15])
        # model output, n*50*12
        cur_y = np.reshape(cur_mat_file['y'], [478*574, 50, 12])

        if each_num == 0:
        # concatenate data
        	all_input = np.concatenate((all_input, cur_x), axis=0)
        	all_output = np.concatenate((all_output, cur_y), axis=0)
        else:
        	all_index = np.arange(478*574)
        	all_index = np.random.permutation(all_index)
        	select_index = all_index[:each_num]
        	all_input = np.concatenate((all_input, cur_x[select_index]), axis=0)
        	all_output = np.concatenate((all_output, cur_y[select_index]), axis=0)
        print('Finish loading ' + file_name)
    # remove the first line of 0s
    all_input = all_input[1:]
    all_output = all_output[1:]

    # input previous
    input_prev = all_input[:, :, :12]

    # use difference as the final model output
    all_output = all_output - input_prev

    # normalize x
    norm_x = x_norm(all_input)

    # normalize outputs
    scale = np.ndarray(shape=(1, 1, 12),
                       buffer=np.array([1e6, 1e7, 1e7, 1e7, 1e7, 1e7, 1e-3, 1, 1e-8, 1e-7, 1e-2, 1e3]))
    norm_y = all_output * scale

    return norm_x, norm_y


# load test dataset for one mat file
# test_data_id: [1, 30]
def generate_test_dataset(test_data_id=27):
    dirs = os.listdir('./all_mp_mat/')
    dirs.sort(key=list_sort)
    file_name = dirs[test_data_id-1]
    cur_mat_file = sci.loadmat('./all_mp_mat/'+file_name)
    # model input, n*50*15
    cur_x = np.reshape(cur_mat_file['x'], [478*574, 50, 15])
    # normalized model input
    norm_x = x_norm(cur_x)
    # previous of input
    prev_x = cur_x[:, :, :12]
    # true y, 478*574*50*12
    true_y = cur_mat_file['y']

    return norm_x, prev_x, true_y


# normalize x, makes it range in [-1, 1]
# return normalized data and its corresponding max and min values
def x_norm(x):
    max_data = np.ndarray(shape=(1, 1, 15), buffer=np.array([2.2886455e-02, 3.6361497e-03, 1.2328358e-03,
                                                            8.4741516e-03, 7.5198379e-03, 5.9695924e-03,
                                                            3.5232464e+07, 1.6976810e+04, 3.3563699e+08,
                                                            1.2960885e+08, 5.8382360e+04, 3.0823670e+02,
                                                            1.0082590e+05, 1.9190690e+01, 5.9400200e+02]))
    min_data = np.ndarray(shape=(1, 1, 15), buffer=np.array([0, 0, 0, 0,
                                                            0, 0, 0, 0,
                                                            0, 9.642865e+06, 4.343633e+03, 1.873498e+02,
                                                            5.182791e+03, -6.290287e+00, 4.794702e+01]))
    diff_data = max_data - min_data
    # if all values are zeros, then make the max - min to be 1
    diff_data[diff_data == 0] = 1
    # normalize input
    norm_data = (x - min_data) / diff_data

    return norm_data


# sort list of file
def list_sort(x):
    x = x.rpartition('_')[0]
    return int(x[4:])
