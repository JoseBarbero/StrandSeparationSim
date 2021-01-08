import numpy as np
import re
import os
import autokeras as ak
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, LayerNormalization
from ReadData import read_data_as_img, read_data_structured, read_data_st, read_data_st_withseq
from Preprocessing import ros, smote, adasyn
from Results import report_results_imagedata, make_spider_by_temp, report_results_st, test_results, plot_train_history
from keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from keras import Sequential
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, log_loss, fowlkes_mallows_score, cohen_kappa_score, precision_score, recall_score
from datetime import datetime
from contextlib import redirect_stdout


def mynet():
    model = keras.Sequential()
    
    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', input_shape=(28,200,1)))
    model.add(MaxPooling2D((2,2)))
    #model.add(Dropout(0.5))

    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    #model.add(Dropout(0.5))

    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu'))
    #model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(units=1024))

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy(), metrics=["accuracy", "AUC"])
    #model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=keras.losses.BinaryCrossentropy(), metrics=["accuracy", "AUC"])

    return model


def channels_net():
    model = keras.Sequential()
    
    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', input_shape=(28,200,2)))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu'))

    model.add(Flatten())

    model.add(Dense(units=1024))

    model.add(Dense(1, activation = 'sigmoid'))

    #model.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy(), metrics=["accuracy", "AUC"])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss=keras.losses.BinaryCrossentropy(), metrics=["accuracy", "AUC"])

    return model

    
if __name__ == "__main__":

    seed = 42
    np.random.seed(seed)

    data_dir = "../data/prueba"
    categories = ["OPN"] #, "BUB10", "BUB12", "BUB8", "VRNORM"]

    X_train, y_train = read_data_st_withseq(data_dir, "train", categories)
    X_val, y_val = read_data_st_withseq(data_dir, "val", categories)
    X_test, y_test = read_data_st_withseq(data_dir, "test", categories)

    #X_train = np.concatenate((X_train, X_val))
    #y_train = np.concatenate((y_train, y_val))
    #X_train, y_train = smote(X_train, y_train)
    #X_train = X_train.reshape(*X_train.shape[:3], 1)
    #X_val = X_val.reshape(*X_val.shape[:3], 1)
    #X_test = X_test.reshape(*X_test.shape[:3], 1)

    if len(sys.argv) < 2:
        run_id = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
        #run_id = "".join(categories)
    else:
        run_id = sys.argv[1]
        #run_id = "".join(categories)

    log_file = "logs/"+run_id+".log"
    hist_file = "logs/"+run_id+".pkl"
    plot_file = "logs/"+run_id+".png"

    model = channels_net()

    with open(log_file, 'w') as f:
        with redirect_stdout(f):
            model.summary()

            for layer in model.layers:
                print(layer.get_config())

            history = model.fit(X_train, y_train,
                                shuffle=True,
                                batch_size=32,
                                epochs=10,
                                verbose=True,
                                validation_data=(X_val, y_val))
            print("Train results:")
            test_results(X_train, y_train, model)
            print("Test results:")
            test_results(X_test, y_test, model)

    with open(hist_file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    plot_train_history(history.history, plot_file)
