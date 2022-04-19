# THIS FILE IS A PART OF NWP MICROPHYSICAL PARAMETERIZATION CODE PROJECT
# Fun03_Experiment.py - train and test the 1DD-CNN regression model, plot training loss, and analyze predicted results
#
# Created on 2022-04-11
# Copyright (c) Ting Shu
# Email: shuting@gbamwf.com

import pickle
import numpy as np
import scipy.io as sci
from numpy import linalg as LA
import matplotlib.pyplot as plt

# train model
def train_model(train_data_set, model,
        batch_size=128, epochs=10, validation_split=0.2,
        save_model=False, model_name=None, save_history=False, his_name=None):
    # ---------------List of Parameters--------------------------#
    # train_data_set: normalized input and output of model
    # model: a build model
    # save_model: boolean, save the trained model or not
    # model_name: string, if save the trained model, provide the name of the saved file
    # save_history: boolean, save the trained history or not
    # his_name: string, if save history, provide the name of the saved file

    # load training input and output
    train_x, train_y = train_data_set

    # train model
    history = model.fit(train_x, train_y,
                        shuffle=True, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=2)
    if save_model:
        model.save(model_name)

    if save_history:
        with open(his_name, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    return model, history


# test model
def test_model(test_data_set, model, do_save=False, res_name=None):
    # ---------------List of Parameters--------------------------#
    # test_data_set: normalized input of model, index of calling microphysical scheme, original input, and true output
    # model: a trained model
    # do_save: boolean, save the predicted result or not
    # res_name: string, if save result, provide the name of the saved file

    # load test data
    test_x, test_prev_x, test_true_y = test_data_set

    # predict test data
    pred_label = model.predict(test_x)

    # reverse the predicted label to the original scale and add previous input
    inverse_scale = np.ndarray(shape=(1, 1, 12),
                               buffer=np.array([1e-6, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e3, 1, 1e8, 1e7, 1e2, 1e-3]))
    pred_label = pred_label * inverse_scale + test_prev_x

    # add previous x and reshape the final predicted matrix
    pred_mat = np.reshape(pred_label, [478, 574, 50, 12])
    if do_save:
        sci.savemat(res_name, {'pred_mat': pred_mat}, do_compression=True)

    return pred_mat, test_true_y


# plot history loss
def plot_loss(his_file, save_plot=False, plot_name=None):
    # ---------------List of Parameters--------------------------#
    # his_file: file of saving history
    # model: a build model
    # save_plot: boolean, save the figure or not
    # plot_name: string, if save this figure, provide the name of the saved file
    with open(his_file, 'rb') as f:
        history = pickle.load(f)

        loss = history['loss']
        val_loss = history['val_loss']
        loss_len = len(loss)
        epochs = range(1, loss_len+1)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        if save_plot:
            plt.savefig(plot_name)


# analyze results
def analyze_results(true_label, pred_label, data_size=600, do_plot=True, var_id=0, level=0, save_plot=False, plot_name=None):
    # ---------------List of Parameters--------------------------#
    # true_label: tensor, 478*574*50*12
    # pred_label: tensor, 478*574*50*12
    # do_plot: boolean, draw compared figures or not
    # var_id: if do_plot=True, provide the id of the variable
    # level: if do_plot=True, provide the level of the variable
    # save_plot: boolean, save the figure or not
    # plot_name: string, if save this figure, provide the name of the saved file
    true2d = np.reshape(true_label, (-1, data_size))
    pred2d = np.reshape(pred_label, (-1, data_size))

    # cosine similarity
    cs_m = np.sum(true2d * pred2d, axis=1)
    true_norm = LA.norm(true2d, axis=1)
    pred_norm = LA.norm(pred2d, axis=1)
    cs = np.mean(cs_m / (true_norm * pred_norm))

    # mean squared error
    mse = np.mean(np.square(pred2d - true2d))

    print('CS and MSE of this model are:' + str(cs) + ' ' + str(mse))

    if do_plot:
        plt.subplot(1, 2, 1)
        plt.pcolor(true_label[:, :, level, var_id])
        plt.subplot(1, 2, 2)
        plt.pcolor(pred_label[:, :, level, var_id])
        plt.show()

        if save_plot:
            plt.savefig(plot_name)

    return cs, mse
