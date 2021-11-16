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
                                    Input, Bidirectional, MultiHeadAttention, LSTM, Add, Concatenate, Average, Maximum, Minimum, Multiply, Dot, Subtract
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
    
    x = tf.keras.layers.MultiHeadAttention(num_heads=32, key_dim=2, attention_axes=(2))(sequence_input, sequence_input)
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


def cnnxlstm09310(seqshape, probsshape):
    
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
    cnn_x = Dense(64, activation = 'relu')(cnn_x)
    cnn_out = Dropout(0.2)(cnn_x)


    merged = Add()([lstm_out, cnn_out])
    z = Dense(128, activation="relu")(merged)
    z = Dense(1, activation="sigmoid")(merged)

    model = tf.keras.Model(inputs=[lstm_in, cnn_in], outputs=z)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", 'AUC'])

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
    cnn_x = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2))(cnn_in, cnn_in)
    cnn_x = Conv2D(filters=32, kernel_size=3, data_format='channels_last', strides=1, activation='relu', input_shape=probsshape)(cnn_in)
    cnn_x = MaxPooling2D(2)(cnn_x)
    cnn_x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(cnn_x)
    cnn_x = MaxPooling2D(2)(cnn_x)
    cnn_x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(cnn_x)
    cnn_x = MaxPooling2D(2)(cnn_x)
    cnn_x = Flatten()(cnn_x)
    cnn_x = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(0))(cnn_x, cnn_x)
    cnn_x = Dense(1024, activation = 'relu')(cnn_x)
    cnn_x = Dropout(0.2)(cnn_x)
    cnn_x = Dense(512, activation = 'relu')(cnn_x)
    cnn_x = Dropout(0.2)(cnn_x)
    cnn_x = Dense(64, activation = 'relu')(cnn_x)
    cnn_out = Dropout(0.2)(cnn_x)

    merged = Multiply()([lstm_out, cnn_out])
    
    z = Dense(128, activation="relu")(merged)
    z = Dense(1, activation="sigmoid")(merged)

    model = tf.keras.Model(inputs=[lstm_in, cnn_in], outputs=z)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", 'AUC'])


    return model



def cnnxlstm_allprobs(seqshape, opn_shape, bub8_shape, bub10_shape, bub12_shape, vrnorm_shape):
    
    lstm_in = Input(shape=seqshape)
    lstm_x = Bidirectional(LSTM(units=64, return_sequences=True, dropout=0.3, input_shape=seqshape))(lstm_in)
    lstm_x = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(1,2))(lstm_x, lstm_x)
    lstm_x = Bidirectional(LSTM(units=64, return_sequences=True, dropout=0.3,))(lstm_x)
    lstm_x = Flatten()(lstm_x)
    lstm_x = Dense(units=64, activation='relu')(lstm_x)
    lstm_out = Dropout(0.75)(lstm_x)
    
    opn_in = Input(shape=opn_shape)
    opn_x = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2))(opn_in, opn_in)
    opn_x = Conv2D(filters=32, kernel_size=3, data_format='channels_last', strides=1, activation='relu')(opn_x)
    opn_x = MaxPooling2D(2)(opn_x)
    opn_x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(opn_x)
    opn_x = MaxPooling2D(2)(opn_x)
    opn_x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(opn_x)
    opn_x = MaxPooling2D(2)(opn_x)
    opn_x = Flatten()(opn_x)
    opn_x = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(0))(opn_x, opn_x)
    opn_x = Dense(1024, activation = 'relu')(opn_x)
    opn_x = Dropout(0.2)(opn_x)
    opn_x = Dense(512, activation = 'relu')(opn_x)
    opn_x = Dropout(0.2)(opn_x)
    opn_x = Dense(64, activation = 'relu')(opn_x)
    opn_out = Dropout(0.2)(opn_x)

    bub8_in = Input(shape=bub8_shape)
    bub8_x = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2))(bub8_in, bub8_in)
    bub8_x = Conv2D(filters=32, kernel_size=3, data_format='channels_last', strides=1, activation='relu')(bub8_x)
    bub8_x = MaxPooling2D(2)(bub8_x)
    bub8_x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(bub8_x)
    bub8_x = MaxPooling2D(2)(bub8_x)
    bub8_x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(bub8_x)
    bub8_x = MaxPooling2D(2)(bub8_x)
    bub8_x = Flatten()(bub8_x)
    bub8_x = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(0))(bub8_x, bub8_x)
    bub8_x = Dense(1024, activation = 'relu')(bub8_x)
    bub8_x = Dropout(0.2)(bub8_x)
    bub8_x = Dense(512, activation = 'relu')(bub8_x)
    bub8_x = Dropout(0.2)(bub8_x)
    bub8_x = Dense(64, activation = 'relu')(bub8_x)
    bub8_out = Dropout(0.2)(bub8_x)

    bub10_in = Input(shape=bub10_shape)
    bub10_x = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2))(bub10_in, bub10_in)
    bub10_x = Conv2D(filters=32, kernel_size=3, data_format='channels_last', strides=1, activation='relu')(bub10_x)
    bub10_x = MaxPooling2D(2)(bub10_x)
    bub10_x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(bub10_x)
    bub10_x = MaxPooling2D(2)(bub10_x)
    bub10_x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(bub10_x)
    bub10_x = MaxPooling2D(2)(bub10_x)
    bub10_x = Flatten()(bub10_x)
    bub10_x = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(0))(bub10_x, bub10_x)
    bub10_x = Dense(1024, activation = 'relu')(bub10_x)
    bub10_x = Dropout(0.2)(bub10_x)
    bub10_x = Dense(512, activation = 'relu')(bub10_x)
    bub10_x = Dropout(0.2)(bub10_x)
    bub10_x = Dense(64, activation = 'relu')(bub10_x)
    bub10_out = Dropout(0.2)(bub10_x)

    bub12_in = Input(shape=bub12_shape)
    bub12_x = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2))(bub12_in, bub12_in)
    bub12_x = Conv2D(filters=32, kernel_size=3, data_format='channels_last', strides=1, activation='relu')(bub12_x)
    bub12_x = MaxPooling2D(2)(bub12_x)
    bub12_x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(bub12_x)
    bub12_x = MaxPooling2D(2)(bub12_x)
    bub12_x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(bub12_x)
    bub12_x = MaxPooling2D(2)(bub12_x)
    bub12_x = Flatten()(bub12_x)
    bub12_x = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(0))(bub12_x, bub12_x)
    bub12_x = Dense(1024, activation = 'relu')(bub12_x)
    bub12_x = Dropout(0.2)(bub12_x)
    bub12_x = Dense(512, activation = 'relu')(bub12_x)
    bub12_x = Dropout(0.2)(bub12_x)
    bub12_x = Dense(64, activation = 'relu')(bub12_x)
    bub12_out = Dropout(0.2)(bub12_x)

    vrnorm_in = Input(shape=vrnorm_shape)
    vrnorm_x = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2))(vrnorm_in, vrnorm_in)
    vrnorm_x = Conv2D(filters=32, kernel_size=3, data_format='channels_last', strides=1, activation='relu')(vrnom_x)
    vrnorm_x = MaxPooling2D(2)(vrnom_x)
    vrnorm_x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(vrnom_x)
    vrnorm_x = MaxPooling2D(2)(vrnom_x)
    vrnorm_x = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu')(vrnom_x)
    vrnorm_x = MaxPooling2D(2)(vrnom_x)
    vrnorm_x = Flatten()(vrnom_x)
    vrnorm_x = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(0))(vrnom_x, vrnom_x)
    vrnorm_x = Dense(1024, activation = 'relu')(vrnom_x)
    vrnorm_x = Dropout(0.2)(vrnom_x)
    vrnorm_x = Dense(512, activation = 'relu')(vrnom_x)
    vrnorm_x = Dropout(0.2)(vrnom_x)
    vrnorm_x = Dense(64, activation = 'relu')(vrnom_x)
    vrnorm_out = Dropout(0.2)(vrnom_x)

    merged = Add()([lstm_out, bub8_out, bub10_out, bub12_out, vrnorm_out])
    
    z = Dense(128, activation="relu")(merged)
    z = Dense(1, activation="sigmoid")(merged)

    model = tf.keras.Model(inputs=[seq_in, opn_in, bub8_in, bub10_in, bub12_in, vrnorm_in], outputs=z)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", 'AUC'])


    return model