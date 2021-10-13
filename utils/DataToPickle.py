import pickle
import numpy as np
from ReadData import *
from datetime import datetime


# Creo este script para no tener que procesar todo cada ejecución y poder guardarlo en unos pocos ficheros más manejables.
data_dir = "../../databubbles/"
categories = ["OPN", "BUB8", "BUB10", "BUB12", "VRNORM"]
#categories = ["BUB8"]
temperatures = ['308.0', '308.3', '308.6', '308.9', 
                '309.2', '309.5', '309.8', 
                '310', '310.1', '310.4', '310.7', 
                '311.0', '311.3', '311.6', '311.9',
                '312.2', '312.5', '312.8',
                '315', '320', '325', '330', '335', '340', '345', '350', '355', '360']

for part in ['train', 'val', 'test']:
    # Train
    print('Reading '+part+' data... '+'['+datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+']')
    
    X, y = read_data_channels_onehot(data_dir, part, temperatures, categories)

    print('Writing '+part+' data... '+'['+datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+']')
    X_file = open('../data/serialized/X_'+part+'.pkl', 'wb')
    pickle.dump(X, X_file, protocol=4)  # protocol=4 allows to serialize larger files than 4gb
    X_file.close()

    y_file = open('../data/serialized/y_'+part+'.pkl', 'wb')
    pickle.dump(y, y_file, protocol=4)
    y_file.close()

    print('Cleaning '+part+' data... '+'['+datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+']')
    del X
    del y