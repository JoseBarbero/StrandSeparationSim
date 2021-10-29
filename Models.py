import sys
sys.path.append("utils")
import numpy as np
import re
import os
import pickle
import tensorflow as tf
from Results import test_results, plot_train_history
from datetime import datetime
from ReadData import get_seq,  get_reversed_seq, get_opn_probs, get_bub8_probs, get_bub10_probs, get_bub12_probs, get_vrnorm_probs
from contextlib import redirect_stdout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Dropout, MaxPooling1D, MaxPooling2D, Flatten, Dense, concatenate, \
                                    Input, Bidirectional, MultiHeadAttention, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler

def lstm(inputshape):

    sequence_input = tf.keras.layers.Input(shape=inputshape)
    
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True, dropout=0.3, input_shape=inputshape))(sequence_input)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.75)(x)
    output = tf.keras.layers.Dense(1)(x)
    output = tf.keras.layers.Activation('sigmoid')(output)

    model = tf.keras.Model(inputs=sequence_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", 'AUC'])

    return model

def att(inputshape):

    sequence_input = tf.keras.layers.Input(shape=inputshape)
    
    x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(1,2))(sequence_input, sequence_input)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.75)(x)
    output = tf.keras.layers.Dense(1)(x)
    output = tf.keras.layers.Activation('sigmoid')(output)

    model = tf.keras.Model(inputs=sequence_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", 'AUC'])

    return model

def lstm_att(inputshape):

    sequence_input = tf.keras.layers.Input(shape=inputshape)
    
    x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=1, attention_axes=(1))(sequence_input, sequence_input)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True, dropout=0.3,))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.75)(x)
    output = tf.keras.layers.Dense(1)(x)
    output = tf.keras.layers.Activation('sigmoid')(output)

    model = tf.keras.Model(inputs=sequence_input, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", 'AUC'])

    return model

def cnn(inputshape):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=3, data_format='channels_last', strides=1, activation='relu', input_shape=inputshape))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'))
    model.add(MaxPooling2D(3))

    model.add(Flatten())

    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(1, activation = 'sigmoid'))

    return model


def cnnxlstm(seqshape, probsshape):
    
    lstm_in = Input(shape=seqshape)
    lstm_x = Bidirectional(LSTM(units=64, return_sequences=True, dropout=0.3, input_shape=seqshape))(lstm_in)
    lstm_x = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(1,2))(lstm_x, lstm_x)
    lstm_x = Bidirectional(LSTM(units=64, return_sequences=True, dropout=0.3,))(lstm_x)
    lstm_x = Flatten()(lstm_x)
    lstm_x = Dense(units=64, activation='relu')(lstm_x)
    lstm_out = Dropout(0.75)(lstm_x)
    
    cnn_in = Input(shape=probsshape)
    cnn_x = Conv2D(filters=32, kernel_size=3, data_format='channels_last', strides=1, activation='relu', input_shape=probsshape)(cnn_in)
    cnn_x = MaxPooling2D(2)(cnn_x)
    cnn_x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(cnn_x)
    cnn_x = MaxPooling2D(2)(cnn_x)
    cnn_x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(cnn_x)
    cnn_x = MaxPooling2D(3)(cnn_x)
    cnn_x = Flatten()(cnn_x)
    cnn_x = Dense(1024, activation = 'relu')(cnn_x)
    cnn_x = Dropout(0.2)(cnn_x)
    cnn_x = Dense(512, activation = 'relu')(cnn_x)
    cnn_x = Dropout(0.2)(cnn_x)
    cnn_x = Dense(128, activation = 'relu')(cnn_x)
    cnn_x = Dropout(0.2)(cnn_x)
    cnn_out = Flatten()(cnn_x)


    merged = concatenate([lstm_out, cnn_out])
    z = Dense(1024, activation="relu")(merged)
    z = Dense(1, activation="sigmoid")(merged)

    model = tf.keras.Model(inputs=[lstm_in, cnn_in], outputs=z)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", 'AUC'])

    return model