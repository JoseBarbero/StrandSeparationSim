import pickle
import numpy as np
import re
import time
from Bio.Seq import Seq
from ReadData import *

data_dir = "../data/prueba"
categories = ["OPN", "BUB8", "BUB10", "BUB12", "VRNORM"]
temperatures = ['308.0', '308.3', '308.6', '308.9', 
                '309.2', '309.5', '309.8', 
                '310', '310.1', '310.4', '310.7', 
                '311.0', '311.3', '311.6', '311.9',
                '312.2', '312.5', '312.8',
                '315', '320', '325', '330', '335', '340', '345', '350', '355', '360']

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

X_train = X_train[:,1,:,5:13]
X_val = X_val[:,1,:,5:13]
X_test = X_test[:,1,:,5:13]


X_train_file = open('../data/serialized/X_train_onlyseq.pkl', 'wb')
pickle.dump(X_train, X_train_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_train_file.close()

y_train_file = open('../data/serialized/y_train_onlyseq.pkl', 'wb')
pickle.dump(y_train, y_train_file, protocol=4)
y_train_file.close()

X_val_file = open('../data/serialized/X_val_onlyseq.pkl', 'wb')
pickle.dump(X_val, X_val_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_val_file.close()

y_val_file = open('../data/serialized/y_val_onlyseq.pkl', 'wb')
pickle.dump(y_val, y_val_file, protocol=4)
y_val_file.close()

X_test_file = open('../data/serialized/X_test_onlyseq.pkl', 'wb')
pickle.dump(X_test, X_test_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_test_file.close()

y_test_file = open('../data/serialized/y_test_onlyseq.pkl', 'wb')
pickle.dump(y_test, y_test_file, protocol=4)
y_test_file.close()
