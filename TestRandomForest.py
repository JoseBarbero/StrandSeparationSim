import numpy as np
import pandas as pd
from ReadData import read_data_as_img, read_data_structured, read_data_st
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from Results import report_results_imagedata, make_spider_by_temp, report_results_st, test_results
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, log_loss, fowlkes_mallows_score, cohen_kappa_score, precision_score, recall_score


if __name__ == "__main__":

    seed = 42
    np.random.seed(seed)

    data_dir = "../data"
    X_train, y_train = read_data_structured(data_dir, "OPNat.*train.*")
    #X_val, y_val = read_data_st(data_dir, "val")
    X_test, y_test = read_data_structured(data_dir, "OPNat.*test.*")

    #X_train = np.concatenate((X_train, X_val))
    #y_train = np.concatenate((y_train, y_val))
    #X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    
    clf = RandomForestClassifier(random_state=seed, verbose=1, n_jobs=8)
    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)

    test_results(X_train, y_train, clf)
    test_results(X_test, y_test, clf)
    