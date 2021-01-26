import pickle
import numpy as np
import re
import time
from Bio.Seq import Seq
from ReadData import *

# Creo este script para no tener que procesar todo cada ejecución y poder guardarlo en unos pocos ficheros más manejables.

data_dir = "../data/prueba"
categories = ["OPN", "BUB8", "BUB10", "BUB12", "VRNORM"]
temperatures = ['308.0', '308.3', '308.6', '308.9', 
                '309.2', '309.5', '309.8', 
                '310', '310.1', '310.4', '310.7', 
                '311.0', '311.3', '311.6', '311.9',
                '312.2', '312.5', '312.8',
                '315', '320', '325', '330', '335', '340', '345', '350', '355', '360']

#X_test, y_test = read_data_channels(data_dir, 'test', temperatures, categories)

#print(np.asarray(seq_to_onehot_array(data_dir+'/onlyseq.TSSnegFineGrained.hg17-test.neg')).shape)
#print(np.asarray(seq_to_onehot_aminoacids(data_dir+'/onlyseq.TSSnegFineGrained.hg17-test.neg')).shape)


X, y = read_data_channels_for_lstmxlstm(data_dir, 'test', temperatures, categories)

print(X.shape)
print(y.shape)