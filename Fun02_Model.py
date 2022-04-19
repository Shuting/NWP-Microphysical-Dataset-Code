# THIS FILE IS A PART OF NWP MICROPHYSICAL PARAMETERIZATION CODE PROJECT
# Fun02_Model.py - build the 1DD-CNN regression model
#
# Created on 2022-04-11
# Copyright (c) Ting Shu
# Email: shuting@gbamwf.com

from tensorflow.keras.layers import Conv1D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def dnn_model(learning_rate=1e-3):
    input = Input(shape=(50, 15))

    x1 = Conv1D(16, 1, activation="relu")(input)
    x1 = Conv1D(16, 3, padding="same", activation="relu")(x1)
    x2 = tf.concat((input, x1), axis=-1)
    x2 = Conv1D(16, 1, activation="relu")(x2)
    x2 = Conv1D(16, 3, padding="same", activation="relu")(x2)
    x3 = tf.concat((input, x1, x2), axis=-1)
    x3 = Conv1D(16, 1, activation="relu")(x3)
    x3 = Conv1D(16, 3, padding="same", activation="relu")(x3)
    x4 = tf.concat((x1, x2, x3), axis=-1)
    x4 = Conv1D(12, 3, padding="same")(x4)
    
    model = Model(inputs=input, outputs=x4)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)

    return model
