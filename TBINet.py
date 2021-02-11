import numpy as np
import re
import os
import pandas as pd
import sys
import pickle
import tensorflow as tf
from ReadData import read_data_as_img, read_data_structured, read_data_st, seq_to_array, seq_to_onehot_array
from Results import report_results_imagedata, make_spider_by_temp, report_results_st, test_results, plot_train_history
from datetime import datetime
from contextlib import redirect_stdout
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def tbinet():
    sequence_input = keras.layers.Input(shape=(200,4))

    # Convolutional Layer
    output = keras.layers.Conv1D(50,kernel_size=3,padding="valid",activation="relu")(sequence_input)
    output = keras.layers.MaxPooling1D(pool_size=3, strides=3)(output)
    output = keras.layers.Dropout(0.2)(output)

    output = keras.layers.Conv1D(32,kernel_size=3,padding="valid",activation="relu")(sequence_input)
    output = keras.layers.MaxPooling1D(pool_size=3, strides=3)(output)
    output = keras.layers.Dropout(0.2)(output)

    output = keras.layers.Conv1D(24,kernel_size=3,padding="valid",activation="relu")(sequence_input)
    output = keras.layers.MaxPooling1D(pool_size=3, strides=3)(output)
    output = keras.layers.Dropout(0.2)(output)

    #Attention Layer
    attention = keras.layers.Dense(1)(output)
    attention = keras.layers.Permute((2, 1))(attention)
    attention = keras.layers.Activation('softmax')(attention)
    attention = keras.layers.Permute((2, 1))(attention)
    attention = keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=2), name='attention',output_shape=(75,))(attention)
    attention = keras.layers.RepeatVector(24)(attention)
    attention = keras.layers.Permute((2,1))(attention)
    output = keras.layers.multiply([output, attention])

    #BiLSTM Layer
    output = keras.layers.Bidirectional(keras.layers.LSTM(320,return_sequences=True))(output)
    output = keras.layers.Dropout(0.5)(output)

    flat_output = keras.layers.Flatten()(output)

    #FC Layer
    FC_output = keras.layers.Dense(1024)(flat_output)
    FC_output = keras.layers.Activation('relu')(FC_output)

    #Output Layer
    output = keras.layers.Dense(1)(FC_output)
    output = keras.layers.Activation('sigmoid')(output)

    model = keras.Model(inputs=sequence_input, outputs=output)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy", "AUC"])

    return model

    
     
if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    X_train_file = open('../data/serialized/X_train_onlyseq.pkl', 'rb')
    y_train_file = open('../data/serialized/y_train_onlyseq.pkl', 'rb')
    X_val_file = open('../data/serialized/X_val_onlyseq.pkl', 'rb')
    y_val_file = open('../data/serialized/y_val_onlyseq.pkl', 'rb')
    X_test_file = open('../data/serialized/X_test_onlyseq.pkl', 'rb')
    y_test_file = open('../data/serialized/y_test_onlyseq.pkl', 'rb')

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

    X_train = X_train[:,:,:4]
    X_val = X_val[:,:,:4]
    X_test = X_test[:,:,:4]
    
    if len(sys.argv) < 2:
        run_id = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
    else:
        run_id = sys.argv[1]
        #run_id = "".join(categories)

    log_file = "logs/"+run_id+".log"
    hist_file = "logs/"+run_id+".pkl"
    plot_file = "logs/"+run_id+".png"

    model = tbinet()

    with open(log_file, 'w') as f:
        with redirect_stdout(f):
            #model.summary()

            for layer in model.layers:
                print(layer.get_config())
            early_stopping_monitor = EarlyStopping( monitor='val_loss', min_delta=0, patience=10, 
                                                    verbose=1, mode='min', baseline=None,
                                                    restore_best_weights=True)
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_delta=1e-4, mode='min')

            history = model.fit(X_train, y_train,
                                shuffle=True,
                                batch_size=32,
                                epochs=50,
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