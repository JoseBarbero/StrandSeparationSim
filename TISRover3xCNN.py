import numpy as np
import re
import os
import autokeras as ak
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import keras
import tensorflow as tf
from Attention import Attention
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AveragePooling2D, LayerNormalization
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D
from keras.layers import LSTM
from keras.layers import concatenate
from ReadData import read_data_as_img, read_data_structured, read_data_st, seq_to_array, seq_to_onehot_array
from Preprocessing import ros, smote, adasyn
from Results import report_results_imagedata, make_spider_by_temp, report_results_st, test_results, plot_train_history
from keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from keras import Sequential
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, log_loss, fowlkes_mallows_score, cohen_kappa_score, precision_score, recall_score
from datetime import datetime
from contextlib import redirect_stdout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

def tisrover():
    model = Sequential()

    model.add(Conv1D(filters=50, kernel_size=3, activation='relu', input_shape=(200, 8)))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters=62, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters=75, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters=87, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters=100, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss=keras.losses.BinaryCrossentropy(), metrics=["accuracy", "AUC"])

    return model

def titerxcnn():
    
    tisrover = Sequential()

    tisrover.add(Conv1D(filters=50, kernel_size=3, activation='relu', input_shape=(200, 8)))
    tisrover.add(Dropout(0.2))

    tisrover.add(Conv1D(filters=62, kernel_size=3, activation='relu'))
    tisrover.add(MaxPooling1D(2))
    tisrover.add(Dropout(0.2))

    tisrover.add(Conv1D(filters=75, kernel_size=3, activation='relu'))
    tisrover.add(MaxPooling1D(2))
    tisrover.add(Dropout(0.2))

    tisrover.add(Conv1D(filters=87, kernel_size=3, activation='relu'))
    tisrover.add(MaxPooling1D(2))
    tisrover.add(Dropout(0.2))

    tisrover.add(Conv1D(filters=100, kernel_size=3, activation='relu'))
    tisrover.add(MaxPooling1D(2))
    tisrover.add(Dropout(0.2))

    tisrover.add(Flatten())

    cnn = Sequential()
    cnn.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu', input_shape=(28,200,5)))
    cnn.add(MaxPooling2D((2,2)))
    cnn.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu'))
    cnn.add(MaxPooling2D((2,2)))
    cnn.add(Conv2D(filters=32, kernel_size=(2, 2), activation='relu'))
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Flatten())

    merged = concatenate([tisrover.output, cnn.output])
    z = Dense(1024, activation="relu")(merged)
    z = Dropout(0.5)(merged)
    z = Dense(1, activation="sigmoid")(merged)

    model = Model(inputs=[tisrover.input, cnn.input], outputs=z)

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', "AUC"])

    return model



    
     
if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    X_train_file = open('../data/serialized/X_train_channels_onehot_noAA.pkl', 'rb')
    y_train_file = open('../data/serialized/y_train_channels_onehot_noAA.pkl', 'rb')
    X_val_file = open('../data/serialized/X_val_channels_onehot_noAA.pkl', 'rb')
    y_val_file = open('../data/serialized/y_val_channels_onehot_noAA.pkl', 'rb')
    X_test_file = open('../data/serialized/X_test_channels_onehot_noAA.pkl', 'rb')
    y_test_file = open('../data/serialized/y_test_channels_onehot_noAA.pkl', 'rb')

    X_train = pickle.load(X_train_file)
    y_train = pickle.load(y_train_file)
    X_val = pickle.load(X_val_file)
    y_val = pickle.load(y_val_file)
    X_test = pickle.load(X_test_file)
    y_test = pickle.load(y_test_file)

    X_train_file.close()
    y_train_file.close()
    X_val_file.close()
    y_val_file.close()
    X_test_file.close()
    y_test_file.close()

    X_train_tisrover = X_train[:,1,:,5:13]
    X_train_cnn = X_train[:,:,:,:5]
    X_val_tisrover = X_val[:,1,:,5:13]
    X_val_cnn = X_val[:,:,:,:5]
    X_test_tisrover = X_test[:,1,:,5:13]
    X_test_cnn = X_test[:,:,:,:5]
    
    if len(sys.argv) < 2:
        run_id = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
    else:
        run_id = sys.argv[1]
        #run_id = "".join(categories)

    log_file = "logs/"+run_id+".log"
    hist_file = "logs/"+run_id+".pkl"
    plot_file = "logs/"+run_id+".png"

    model = tisrover()

    with open(log_file, 'w') as f:
        with redirect_stdout(f):
            #model.summary()

            for layer in model.layers:
                print(layer.get_config())
            early_stopping_monitor = EarlyStopping( monitor='val_loss', min_delta=0, patience=10, 
                                                    verbose=1, mode='min', baseline=None,
                                                    restore_best_weights=True)
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_delta=1e-4, mode='min')

            history = model.fit((X_train_tisrover, X_train_cnn), y_train,
                                shuffle=True,
                                batch_size=32,
                                epochs=50,
                                verbose=True,
                                validation_data=((X_val_tisrover, X_val_cnn), y_val),
                                callbacks=[early_stopping_monitor, reduce_lr_loss])
            print("Train results:")
            test_results((X_train_tisrover, X_train_cnn), y_train, model)
            print("Test results:")
            test_results((X_test_tisrover, X_test_cnn), y_test, model)

    with open(hist_file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    plot_train_history(history.history, plot_file)