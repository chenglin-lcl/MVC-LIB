import numpy as np

def l2_normlize_data(X):
    Q = np.diag(1 / np.sqrt(np.sum(X ** 2, axis=0)))
    X = X @ Q
    return X