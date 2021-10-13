import sys
sys.path.append("utils")
import numpy as np
import re
import os
import pickle
from Results import test_results, plot_train_history
from datetime import datetime
from ReadData import get_seq,  get_reversed_seq, get_opn_probs, get_bub8_probs, get_bub10_probs, get_bub12_probs, get_vrnorm_probs
from contextlib import redirect_stdout
import keras
import tensorflow as tf
from keras import layers
from keras.models import Sequential, Model
from keras.layers import Conv1D, Conv2D, Conv3D, Dropout, MaxPooling1D, MaxPooling2D, Flatten, Dense, concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler

def lstm_att(inputshape):

    sequence_input = tf.keras.layers.Input(shape=inputshape)
    
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True, dropout=0.3, input_shape=inputshape))(sequence_input)
    x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(1,2))(x, x)
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

    #model.add(Conv2D(filters=32, kernel_size=3, data_format='channels_last', strides=1, activation='relu', input_shape=(28, 200, 1)))
    #model.add(Conv2D(filters=32, kernel_size=3, data_format='channels_last', strides=1, activation='relu', input_shape=(200, 4)))
    model.add(Conv1D(filters=32, kernel_size=3, data_format='channels_last', strides=1, activation='relu', input_shape=inputshape))
    
    #model.add(MaxPooling2D(2))

    #model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu'))
    
    #model.add(MaxPooling2D(2))

    #model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'))
    model.add(Conv1D(filters=32, kernel_size=3, strides=1, activation='relu'))
    
    #model.add(MaxPooling2D(3))

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


def cnnxlstm(seqshape, inputshape):
    
    sequence_input = tf.keras.layers.Input(shape=seqshape)
    
    lstm = Sequential() 
    lstm.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True, dropout=0.3, input_shape=seqshape)))
    lstm.add(tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(1,2)))
    lstm.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=True, dropout=0.3,)))
    lstm.add(Flatten())
    

    cnn = Sequential()
    cnn.add(Conv2D(filters=32, kernel_size=3, data_format='channels_last', strides=1, activation='relu', input_shape=inputshape))
    cnn.add(MaxPooling2D(2))
    cnn.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'))
    cnn.add(MaxPooling2D(2))
    cnn.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'))
    cnn.add(MaxPooling2D(3))
    cnn.add(Flatten())
    cnn.add(Dense(1024, activation = 'relu'))
    cnn.add(Dropout(0.2))
    cnn.add(Dense(512, activation = 'relu'))
    cnn.add(Dropout(0.2))
    cnn.add(Dense(128, activation = 'relu'))
    cnn.add(Dropout(0.2))
    cnn.add(Flatten())


    merged = concatenate([lstm.output, cnn.output])
    z = Dense(1024, activation="relu")(merged)
    z = Dense(1, activation="sigmoid")(merged)

    model = Model(inputs=[lstm.input, cnn.input], outputs=z)

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', "AUC"])

    return model