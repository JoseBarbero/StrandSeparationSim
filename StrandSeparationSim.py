import numpy as np
from ReadData import read_data_as_img, read_data_structured
from Models import lenet, alexnet, widenet
from keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import autokeras as ak
from sklearn import svm

if __name__ == "__main__":

    seed = 42
    np.random.seed(seed)

    data_dir = "../data"
    X_train, y_train = read_data_structured(data_dir, "OPNat.*train.*")
    X_val, y_val = read_data_structured(data_dir, "OPNat.*val.*")
    X_test, y_test = read_data_structured(data_dir, "OPNat.*test.*")

    clf_sub345 = svm.SVC()
    clf_sub345.fit(X_train, y_train)