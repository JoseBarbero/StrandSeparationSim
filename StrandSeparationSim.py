import numpy as np
import re
import os
import autokeras as ak
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, LayerNormalization
from ReadData import read_data_as_img, read_data_structured, read_data_st
from Preprocessing import ros, smote, adasyn
from Results import report_results_imagedata, make_spider_by_temp, report_results_st, test_results
from keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from keras import Sequential
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, log_loss, fowlkes_mallows_score, cohen_kappa_score, precision_score, recall_score
from datetime import datetime
from contextlib import redirect_stdout


def widenet():
    model = keras.Sequential()
    
    model.add(Conv2D(filters=100, kernel_size=(3, 3), activation='relu', input_shape=(11,200,1)))
    
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    model.add(Dropout(0.5))

    model.add(Dense(units=128, activation='relu'))

    model.add(Dense(1, activation = 'sigmoid'))

    
    # Compile the model
    # SGD
    # RMSprop
    # Adam
    # Adadelta
    # Adagrad
    # Adamax
    # Nadam
    # Ftrl
    model.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

    return model

    
if __name__ == "__main__":

    seed = 42
    np.random.seed(seed)

    data_dir = "../data"
    X_train, y_train = read_data_st(data_dir, "train")
    X_val, y_val = read_data_st(data_dir, "val")
    X_test, y_test = read_data_st(data_dir, "test")

    X_train = np.concatenate((X_train, X_val))
    y_train = np.concatenate((y_train, y_val))
    #X_train, y_train = smote(X_train, y_train)
    X_train = X_train.reshape(*X_train.shape[:3], 1)
    X_test = X_test.reshape(*X_test.shape[:3], 1)

    logfile = "logs/"+str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]+".log"

    model = widenet()

    with open(logfile, 'w') as f:
        with redirect_stdout(f):
            model.summary()

            for layer in model.layers:
                print(layer.get_config())

            model.fit(X_train, y_train,
                        shuffle=True,
                        batch_size=32,
                        epochs=100,
                        verbose=True,
                        validation_split=0.2)

            test_results(X_train, y_train, model)
            test_results(X_test, y_test, model)
