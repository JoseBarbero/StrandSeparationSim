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
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, LayerNormalization
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

def lstm():
    
    model = Sequential() 
    model.add(LSTM(32, return_sequences=True, go_backwards=True, input_shape=(200, 5)))
    #lstm_seq.add(Dropout(0.5))
    model.add(Attention(name='attention_seq'))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))


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


    if len(sys.argv) < 2:
        run_id = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
    else:
        run_id = sys.argv[1]
        #run_id = "".join(categories)

    log_file = "logs/"+run_id+".log"
    hist_file = "logs/"+run_id+".pkl"
    plot_file = "logs/"+run_id+".png"

    '''
    X_train_opn = X_train[:, 0].reshape((*X_train[:, 0].shape, 1))
    print(X_train_opn.shape)
    #X_train_bub8 = X_train[:, 1].reshape((*X_train[:, 1].shape, 1))
    #X_train_bub10 = X_train[:, 2].reshape((*X_train[:, 2].shape, 1))
    #X_train_bub12 = X_train[:, 3].reshape((*X_train[:, 3].shape, 1))
    #X_train_vrnorm = X_train[:, 4].reshape((*X_train[:, 4].shape, 1))
    X_train_seq = X_train[:, 5:9]
    #X_train_seq_comp = X_train[:, 9:13]
    
    X_val_opn = X_val[:, 0].reshape((*X_val[:, 0].shape, 1))
    #X_val_bub8 = X_val[:, 1].reshape((*X_val[:, 1].shape, 1))
    #X_val_bub10 = X_val[:, 2].reshape((*X_val[:, 2].shape, 1))
    #X_val_bub12 = X_val[:, 3].reshape((*X_val[:, 3].shape, 1))
    #X_val_vrnorm = X_val[:, 4].reshape((*X_val[:, 4].shape, 1))
    X_val_seq = X_val[:, 5:9]
    #X_val_seq_comp = X_val[:, 9:13]
    
    X_test_opn = X_test[:, 0].reshape((*X_test[:, 0].shape, 1))
    #X_test_bub8 = X_test[:, 1].reshape((*X_test[:, 1].shape, 1))
    #X_test_bub10 = X_test[:, 2].reshape((*X_test[:, 2].shape, 1))
    #X_test_bub12 = X_test[:, 3].reshape((*X_test[:, 3].shape, 1))
    #X_test_vrnorm = X_test[:, 4].reshape((*X_test[:, 4].shape, 1))
    X_test_seq = X_test[:, 5:9]
    #X_test_seq_comp = X_test[:, 9:13]
    '''
    print(X_train.shape)
    X_train = np.concatenate((X_train[:,:,5:9], X_train[:,:,0, None]), axis=2)
    
    X_val = np.concatenate((X_val[:,:,5:9], X_val[:,:,0, None]), axis=2)
    
    X_test = np.concatenate((X_test[:,:,5:9], X_test[:,:,0, None]), axis=2)

    print(X_train.shape)


    model = lstm()

    with open(log_file, 'w') as f:
        with redirect_stdout(f):
            #model.summary()

            for layer in model.layers:
                print(layer.get_config())
            early_stopping_monitor = EarlyStopping( monitor='val_loss', min_delta=0, patience=6, 
                                                    verbose=1, mode='min', baseline=None,
                                                    restore_best_weights=True)
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_delta=1e-4, mode='min')

            history = model.fit(X_train, y_train,
                                shuffle=True,
                                batch_size=32,
                                epochs=100,
                                verbose=True,
                                validation_data=(X_val, y_val),
                                callbacks=[early_stopping_monitor, reduce_lr_loss])
            print("Train results:")
            test_results(X_train, y_train, model)
            
            print("Test results:")

            test_results(X_test, y_test, model)

    with open(hist_file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    plot_train_history(history.history, plot_file)