import numpy as np
from sklearn.cluster import KMeans

'''
Initialization method for NMF
X [m, n]
U [m, n_components]
V [n_components, n]
m: feature dimension
n: sample number
n_components: subspace dimension
'''


def init_by_kmeans(X, n_components):
    n = X.shape[1]
    y_pred = KMeans(n_clusters=n_components, n_init=10).fit_predict(X.T)  # n*m
    # label 
    label = list(y_pred)
    label_trans = np.zeros_like(label, dtype=np.int32)
    # Remove duplicate elements without changing the order
    class_label = list(set(label))
    class_label.sort(key=label.index)
    # Adjust the index of the cluster
    for i in range(len(class_label)):
        label_trans[label == class_label[i]] = i
    # init V
    V = np.zeros([n_components, n], dtype=np.float32)
    x_pos = label_trans
    y_pos = [i for i in range(len(label_trans))]
    pos = tuple([x_pos, y_pos])
    V[pos] = 1
    V = np.maximum(V, 0.01)
    # init U
    U = X @ V.T

    return U, V


def random_init(X, n_components):
    m, n = X.shape
    # init U
    U = np.abs(np.random.rand(m, n_components))
    # init V
    V = np.abs(np.random.rand(n_components, n))
    return U, V


