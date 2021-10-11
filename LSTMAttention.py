import sys
sys.path.append("utils")
import numpy as np
import re
import os
import pickle
from Results import test_results, plot_train_history
from datetime import datetime
from contextlib import redirect_stdout
import keras
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.layers import Conv1D, Conv2D, Conv3D, Dropout, MaxPooling1D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler

def lstm_att():
    model = keras.models.Sequential()
    
    # Esto así tal cual sobreajusta mucho
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=64, return_sequences=True, dropout=0.3, input_shape=(200,4))))
    #model.add(keras.layers.Dropout(0.75))
    model.add(keras.layers.MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(1,2)))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=64, return_sequences=True, dropout=0.3,)))
    #model.add(keras.layers.Dropout(0.75))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=64, activation='relu'))
    model.add(keras.layers.Dropout(0.75))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))

    #model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=["accuracy", 'AUC'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", 'AUC'])

    return model


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
            model.summary()

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

        X_train_file = open('../databubbles/serialized/X_train_onlyseq.pkl', 'rb')
        y_train_file = open('../databubbles/serialized/y_train_onlyseq.pkl', 'rb')
        X_val_file = open('../databubbles/serialized/X_val_onlyseq.pkl', 'rb')
        y_val_file = open('../databubbles/serialized/y_val_onlyseq.pkl', 'rb')
        X_test_file = open('../databubbles/serialized/X_test_onlyseq.pkl', 'rb')
        y_test_file = open('../databubbles/serialized/y_test_onlyseq.pkl', 'rb')

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

        X_train = X_train[:,:,:4]
        X_val = X_val[:,:,:4]
        X_test = X_test[:,:,:4]

        y_train = y_train.reshape(*y_train.shape, 1)
        y_val = y_val.reshape(*y_val.shape, 1)
        y_test = y_test.reshape(*y_test.shape, 1)
        
        if len(sys.argv) < 2:
            run_id = str(datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_").split(".")[0]
        else:
            run_id = sys.argv[1]
            #run_id = "".join(categories)

        
        single_train(lstm_att(), X_train, X_val, X_test, y_train, y_val, y_test, run_id)
