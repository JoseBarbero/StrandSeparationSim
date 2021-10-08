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

def tisrover3xtisrover3():
    
    seq_input = layers.Input(shape=(200,8))
    
    x = layers.Conv1D(filters=50, kernel_size=3, activation='relu')(seq_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(filters=62, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(filters=75, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(filters=87, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(filters=100, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=8)(x,x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.Dropout(0.5)(x)
    seq_output = layers.Flatten()(x)

    probs = keras.models.Sequential()
    probs.add(layers.Conv2D(filters=50, kernel_size=(2, 2), activation='relu', input_shape=(28,200,5)))
    probs.add(layers.Dropout(0.2))
    probs.add(layers.Conv2D(filters=62, kernel_size=(2, 2), activation='relu'))
    probs.add(layers.MaxPooling2D((2,2)))
    probs.add(layers.Dropout(0.2))
    probs.add(layers.Conv2D(filters=75, kernel_size=(2, 2), activation='relu'))
    probs.add(layers.MaxPooling2D((2,2)))
    probs.add(layers.Dropout(0.2))
    probs.add(layers.Conv2D(filters=87, kernel_size=(2, 2), activation='relu'))
    probs.add(layers.MaxPooling2D((2,2)))
    probs.add(layers.Dropout(0.2))
    probs.add(layers.Conv1D(filters=100, kernel_size=2, activation='relu'))
    probs.add(layers.MaxPooling2D((2,2)))
    probs.add(layers.Dropout(0.2))
    probs.add(layers.Flatten())
    probs.add(layers.Dense(128, activation='relu'))
    probs.add(layers.Dropout(0.5))
    probs.add(layers.Flatten())
    
    merged = layers.concatenate([seq_output, probs.output])
    z = layers.Dense(1024, activation="relu")(merged)
    z = layers.Dropout(0.5)(merged)
    z = layers.Dense(1, activation="sigmoid")(merged)

    model = keras.Model(inputs=[seq_input, probs.input], outputs=z)

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

    model = tisrover3xtisrover3()

    with open(log_file, 'w') as f:
        with redirect_stdout(f):
            #model.summary()

            for layer in model.layers:
                print(layer.get_config())
            early_stopping_monitor = EarlyStopping( monitor='val_loss', min_delta=0, patience=10, 
                                                    verbose=1, mode='min', baseline=None,
                                                    restore_best_weights=True)
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_delta=1e-4, mode='min')

            history = model.fit([X_train_tisrover, X_train_cnn], y_train,
                                shuffle=True,
                                batch_size=32,
                                epochs=50,
                                verbose=True,
                                validation_data=([X_val_tisrover, X_val_cnn], y_val),
                                callbacks=[early_stopping_monitor, reduce_lr_loss])
            print("Train results:")
            test_results([X_train_tisrover, X_train_cnn], y_train, model)
            print("Test results:")
            test_results([X_test_tisrover, X_test_cnn], y_test, model)

    with open(hist_file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    plot_train_history(history.history, plot_file)