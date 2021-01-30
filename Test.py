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

X_test_file = open('../data/serialized/X_test_channels_onehot_noAA.pkl', 'rb')
y_test_file = open('../data/serialized/y_test_channels_onehot_noAA.pkl', 'rb')

X_test = pickle.load(X_test_file)
y_test = pickle.load(y_test_file)

X_test_file.close()
y_test_file.close()

print(X_test.shape)
X_test_seq = X_test[:,1,:,5:13]
print(X_test_seq.shape)
# La idea aquí es apilar las temperaturas en el último eje, como un canal más en vez de como lineas de una imagen
X_test_probs = np.moveaxis(X_test[:,:,:,0], 1, 2) 

print(X_test_probs.shape)

X_test = np.concatenate((X_test_seq, X_test_probs), axis=2)
print(X_test.shape)