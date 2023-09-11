import numpy as np
from utils import init_by_kmeans, random_init
from sklearn.preprocessing import normalize
from utils import Graph


class NMF(object):
    """
    NMF

    example:
    V_star, obj_val, cnt = NMF(n_components=class_num, norm='fro').fit(X)
    X: m*n, V_star: n_components*n

    Reference:

    [1] Lee D D, Seung H S. Learning the parts of objects by non-negative matrix factorization[J]. Nature, 1999, 401(6755): 788-791.
    [2] Kong D, Ding C, Huang H. Robust nonnegative matrix factorization using l21-norm[C]
        Proceedings of the 20th ACM international conference on Information and knowledge management. 2011: 673-682.
    [3] Wang Q, He X, Jiang X, et al. Robust bi-stochastic graph regularized matrix factorization for data clustering
        [J].IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020, 44(1): 390-403.
    """

    def __init__(self, n_components, norm, max_iter=1000, tol=1e-5, kmeans_init=False) -> None:
        self.n_components = n_components
        self.kmeans_init = kmeans_init
        self.max_iter = max_iter
        self.tol = tol
        assert norm in ['fro', 'l21', 'log'], "norm is from ['fro', 'l21', 'log']"
        self.norm = norm

    def init_W_H(self, X):
        n_components = self.n_components
        W = None
        H = None
        if self.kmeans_init:
            # Use kmeans for initialization
            W, H = init_by_kmeans(X, n_components)
        else:
            # random
            W, H = random_init(X, n_components)
        return W, H

    def calc_obj(self, X, W, H):
        obj = 0
        if self.norm == 'fro':
            obj = ((np.linalg.norm(X - W @ H, ord='fro')) ** 2)
        elif self.norm == 'l21':
            obj = np.sum(np.sqrt(np.sum(((X - W @ H) ** 2), axis=0)))
        elif self.norm == 'log':
            obj = np.sum(np.log(1 + np.sqrt(np.sum(((X - W @ H) ** 2), axis=0))))
        return obj

    def get_sample_weight(self, X, W, H):
        m, n = X.shape
        D = None
        if self.norm == 'fro':
            D = np.eye(n)
        elif self.norm == 'l21':
            E = X - W @ H
            Ei = np.maximum(np.sqrt(np.sum(E * E, axis=0)), 1e-9)
            D = np.diag(1. / (2 * Ei))
        elif self.norm == 'log':
            E = X - W @ H
            Ei = np.maximum(np.sqrt(np.sum(E * E, axis=0)), 1e-9)
            D = np.diag(1. / (2 * Ei * (1 + Ei)))
        return D

    def fit(self, X):
        X = X.copy()
        max_iter = self.max_iter
        tol = self.tol
        obj = []
        error_cnt = 0
        W, H = self.init_W_H(X)

        # update
        for t in range(max_iter):

            # update H
            D = self.get_sample_weight(X, W, H)
            tempUp = W.T @ X @ D
            tempDown = W.T @ W @ H @ D
            H = H * (tempUp / np.maximum(tempDown, 1e-9))

            # update W
            tempUp = X @ D @ H.T
            tempDown = W @ H @ D @ H.T
            W = W * (tempUp / np.maximum(tempDown, 1e-9))

            obj.append(self.calc_obj(X, W, H))
            print('iter = {}, obj = {:.2f}'.format(t + 1, obj[t]))

            if t > 0 and obj[t] > obj[t - 1]:
                error_cnt += 1
            if t > 0 and np.abs(obj[t] - obj[t - 1]) / obj[t - 1] <= tol:
                break

        return H, obj, error_cnt


class GNMF(object):
    """
    GNMF

    example:
    V_star, obj_val, cnt = GNMF(n_components=class_num, alpha=0.001).fit(X)
    X: m*n, V_star: n_components*n

    Reference:

    [1] Cai D, He X, Han J, et al. Graph regularized nonnegative matrix factorization for data representation
        [J]. IEEE transactions on pattern analysis and machine intelligence, 2010, 33(8): 1548-1560.
    """

    def __init__(self, n_components, alpha, max_iter=1000, tol=1e-5, kmeans_init=False) -> None:
        self.n_components = n_components
        self.kmeans_init = kmeans_init
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha

    def init_W_H(self, X):
        n_components = self.n_components
        W = None
        H = None
        if self.kmeans_init:
            # Use kmeans for initialization
            W, H = init_by_kmeans(X, n_components)
        else:
            # random
            W, H = random_init(X, n_components)
        return W, H

    def calc_obj(self, X, W, H, L):
        obj = ((np.linalg.norm(X - W @ H, ord='fro')) ** 2) + self.alpha * np.trace(H @ L @ H.T)
        return obj

    def fit(self, X):
        X = X.copy()
        max_iter = self.max_iter
        tol = self.tol
        alpha = self.alpha
        obj = []
        error_cnt = 0
        # init WH
        W, H = self.init_W_H(X)
        # init graph
        S = Graph(neighbor_num=5).get_adjacency_matrix(X)
        D = np.diag(np.sum(S, axis=0))
        L = D - S

        # update
        for t in range(max_iter):

            # update H
            tempUp = W.T @ X + alpha * H @ S
            tempDown = W.T @ W @ H + alpha * H @ D
            H = H * (tempUp / np.maximum(tempDown, 1e-9))

            # update W
            tempUp = X @ H.T
            tempDown = W @ H @ H.T
            W = W * (tempUp / np.maximum(tempDown, 1e-9))

            obj.append(self.calc_obj(X, W, H, L))
            print('iter = {}, obj = {:.2f}'.format(t + 1, obj[t]))

            if t > 0 and obj[t] > obj[t - 1]:
                error_cnt += 1
            if t > 0 and np.abs(obj[t] - obj[t - 1]) / obj[t - 1] <= tol:
                break

        return H, obj, error_cnt
