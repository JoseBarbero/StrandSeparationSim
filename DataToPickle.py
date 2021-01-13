import pickle
import numpy as np
from ReadData import read_data_st_withseq_onehot, read_data_channels

# Creo este script para no tener que procesar todo cada ejecución y poder guardarlo en unos pocos ficheros más manejables.
data_dir = "../data/prueba"
categories = ["OPN", "BUB8", "BUB10", "BUB12", "VRNORM"]
temperatures = ['308.0', '308.3', '308.6', '308.9', 
                '309.2', '309.5', '309.8', 
                '310', '310.1', '310.4', '310.7', 
                '311.0', '311.3', '311.6', '311.9',
                '312.2', '312.5', '312.8',
                '315', '320', '325', '330', '335', '340', '345', '350', '355', '360']

# Train
X_train, y_train = read_data_channels(data_dir, "train", temperatures, categories)

X_train_file = open('../data/serialized/X_train_channels.pkl', 'wb')
pickle.dump(X_train, X_train_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_train_file.close()

y_train_file = open('../data/serialized/y_train_channels.pkl', 'wb')
pickle.dump(y_train, y_train_file,protocol=4)
y_train_file.close()


# Val
X_val, y_val = read_data_channels(data_dir, "val", temperatures, categories)

X_val_file = open('../data/serialized/X_val_channels.pkl', 'wb')
pickle.dump(X_val, X_val_file, protocol=4)
X_val_file.close()

y_val_file = open('../data/serialized/y_val_channels.pkl', 'wb')
pickle.dump(y_val, y_val_file, protocol=4)
y_val_file.close()


# Test
X_test, y_test = read_data_channels(data_dir, "test", temperatures, categories)

X_test_file = open('../data/serialized/X_test_channels.pkl', 'wb')
pickle.dump(X_test, X_test_file, protocol=4)
X_test_file.close()

y_test_file = open('../data/serialized/y_test_channels.pkl', 'wb')
pickle.dump(y_test, y_test_file, protocol=4)
y_test_file.close()





