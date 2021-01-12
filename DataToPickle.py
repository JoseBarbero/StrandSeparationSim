import pickle
import numpy as np
from ReadData import read_data_st_withseq_onehot

# Creo este script para no tener que procesar todo cada ejecución y poder guardarlo en unos pocos ficheros más manejables.

data_dir = "../data/prueba"
categories = ["OPN" ] #, "BUB10", "BUB12", "BUB8", "VRNORM"]


#X_train = np.concatenate((X_train, X_val))
#y_train = np.concatenate((y_train, y_val))
#X_train, y_train = smote(X_train, y_train)
#X_train = X_train.reshape(*X_train.shape[:3], 1)
#X_val = X_val.reshape(*X_val.shape[:3], 1)
#X_test = X_test.reshape(*X_test.shape[:3], 1)

X_train, y_train = read_data_st_withseq_onehot(data_dir, "train", categories)

X_train_file = open('../data/serialized/X_train_withseqs_onehot_OPN.pkl', 'wb')
pickle.dump(X_train, X_train_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
X_train_file.close()

y_train_file = open('../data/serialized/y_train_withseqs_onehot_OPN.pkl', 'wb')
pickle.dump(y_train, y_train_file,protocol=4)
y_train_file.close()


X_val, y_val = read_data_st_withseq_onehot(data_dir, "val", categories)

X_val_file = open('../data/serialized/X_val_withseqs_onehot_OPN.pkl', 'wb')
pickle.dump(X_val, X_val_file, protocol=4)
X_val_file.close()

y_val_file = open('../data/serialized/y_val_withseqs_onehot_OPN.pkl', 'wb')
pickle.dump(y_val, y_val_file, protocol=4)
y_val_file.close()


X_test, y_test = read_data_st_withseq_onehot(data_dir, "test", categories)

X_test_file = open('../data/serialized/X_test_withseqs_onehot_OPN.pkl', 'wb')
pickle.dump(X_test, X_test_file, protocol=4)
X_test_file.close()

y_test_file = open('../data/serialized/y_test_withseqs_onehot_OPN.pkl', 'wb')
pickle.dump(y_test, y_test_file, protocol=4)
y_test_file.close()





