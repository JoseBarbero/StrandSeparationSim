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
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, Conv3D, Dropout, MaxPooling1D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler
from Models import lstm_att, cnn

def single_train(model_definition, X_train, X_val, X_test, y_train, y_val, y_test, run_id):

    log_file = "logs/"+run_id+".log"
    hist_file = "logs/"+run_id+".pkl"
    plot_file = "logs/"+run_id+".png"
    model_file = "logs/"+run_id+".h5"

    logdir = os.path.dirname(log_file)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    model = model_definition
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", 'AUC'])
    

    with open(log_file, 'w') as f:
        with redirect_stdout(f):

            for layer in model.layers:
                print(layer.get_config())
            early_stopping_monitor = EarlyStopping( monitor='val_loss', min_delta=0, patience=10, 
                                                    verbose=1, mode='min', baseline=None,
                                                    restore_best_weights=True)
            reduce_lr_loss = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, verbose=1, min_delta=1e-4, mode='max')

            history = model.fit(X_train, y_train,
                                shuffle=True,
                                batch_size=32,
                                epochs=100,
                                verbose=True,
                                validation_data=(X_val, y_val),
                                callbacks=[early_stopping_monitor, reduce_lr_loss])
            print("Train results:\n")
            test_results(X_train, y_train, model)
            print("Val results:\n")
            test_results(X_val, y_val, model)
            print("Test results:\n")
            test_results(X_test, y_test, model)
            

    with open(hist_file, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    model.save(model_file)
    plot_train_history(history.history, plot_file)



if __name__ == "__main__":
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2"])
    with strategy.scope():
        seed = 42
        np.random.seed(seed)

        X_train_file = open('../databubbles/serialized/X_train.pkl', 'rb')
        y_train_file = open('../databubbles/serialized/y_train.pkl', 'rb')
        X_val_file = open('../databubbles/serialized/X_val.pkl', 'rb')
        y_val_file = open('../databubbles/serialized/y_val.pkl', 'rb')
        X_test_file = open('../databubbles/serialized/X_test.pkl', 'rb')
        y_test_file = open('../databubbles/serialized/y_test.pkl', 'rb')

        X_train = pickle.load(X_train_file)
        #X_train = np.reshape(X_train, (*X_train.shape, 1))
        y_train = pickle.load(y_train_file)
        X_val = pickle.load(X_val_file)
        #X_val = np.reshape(X_val, (*X_val.shape, 1))
        y_val = pickle.load(y_val_file)
        X_test = pickle.load(X_test_file)
        #X_test = np.reshape(X_test, (*X_test.shape, 1))
        y_test = pickle.load(y_test_file)

        X_train_file.close()
        y_train_file.close()
        X_val_file.close()
        y_val_file.close()
        X_test_file.close()
        y_test_file.close()
        
        print('print(X_train.shape)', X_train.shape)
        print('print(get_seq.shape)', get_seq(X_train).shape)
        print('print(get_bub8_probs.shape)', get_bub8_probs(X_train).shape)
        X_train = np.concatenate((get_seq(X_train), get_bub8_probs(X_train)), axis=3)
        print(X_train.shape)
        X_val = np.concatenate((get_seq(X_val), get_bub8_probs(X_val)), axis=3)
        X_test = np.concatenate((get_seq(X_test), get_bub8_probs(X_test)), axis=3)

        X_train = X_train.astype(int)
        X_val = X_val.astype(int)
        X_test = X_test.astype(int)

        y_train = y_train.reshape(*y_train.shape, 1)
        y_val = y_val.reshape(*y_val.shape, 1)
        y_test = y_test.reshape(*y_test.shape, 1)
        
        if len(sys.argv) < 2:
            run_id = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
        else:
            run_id = sys.argv[1]

        
        single_train(lstm_att(X_train.shape[1:]), X_train, X_val, X_test, y_train, y_val, y_test, run_id)