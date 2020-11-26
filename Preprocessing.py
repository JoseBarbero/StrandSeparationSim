import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN


def quick_random_undersampler(X, y):
    """
    Esto es de usar y tirar.
    """
    new_X = []
    new_y = []
    np.random.seed(42)
    class_0_count = np.count_nonzero(y == 0)
    class_1_count = np.count_nonzero(y == 1)
    proportion = class_1_count / class_0_count
    for i in range(y.shape[0]):
        if y[i] == 0:
            if np.random.rand() < proportion:
                new_X.append(X[i])
                new_y.append(y[i])
        else:
            new_X.append(X[i])
            new_y.append(y[i])
    return (np.asarray(new_X), np.asarray(new_y))

def quick_random_oversampler(X, y):
    """
    Esto es de usar y tirar.
    """
    new_X = []
    new_y = []
    np.random.seed(42)
    class_0_count = np.count_nonzero(y == 0)
    class_1_count = np.count_nonzero(y == 1)
    proportion = class_1_count / class_0_count
    for i in range(y.shape[0]):
        new_X.append(X[i])
        new_y.append(y[i])
        if y[i] == 1:
            for j in range(int(proportion*100)):
                new_X.append(X[i])
                new_y.append(y[i])
            
    return (np.asarray(new_X), np.asarray(new_y))

def ros(X, y):
    orig_shape = X.shape
    X = X.reshape(X.shape[0], -1)
    
    ros = RandomOverSampler(random_state=42)
    
    X_resampled, y_resampled = ros.fit_resample(X, y)
    X_resampled = X_resampled.reshape(X_resampled.shape[0], orig_shape[1], orig_shape[2])
    
    return(X_resampled, y_resampled)

def smote(X, y):
    orig_shape = X.shape
    X = X.reshape(X.shape[0], -1)
    
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    
    X_resampled = X_resampled.reshape(X_resampled.shape[0], orig_shape[1], orig_shape[2])
    
    return(X_resampled, y_resampled)
    

def adasyn(X, y):
    orig_shape = X.shape
    X = X.reshape(X.shape[0], -1)
    
    X_resampled, y_resampled = ADASYN().fit_resample(X, y)
    
    X_resampled = X_resampled.reshape(X_resampled.shape[0], orig_shape[1], orig_shape[2])
    
    return(X_resampled, y_resampled)


def check_n_instances(prev_y, new_y):

    print(f"Original class 0 instances: {np.count_nonzero(prev_y == 0)}")
    print(f"Original class 1 instances: {np.count_nonzero(prev_y == 1)}")

    print(f"Current class 0 instances: {np.count_nonzero(new_y == 0)}")
    print(f"Current class 1 instances: {np.count_nonzero(new_y == 1)}")