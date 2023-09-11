import numpy as np
from utils import init_by_kmeans, random_init
from sklearn.preprocessing import normalize


class DiNMF(object):
    """
    multi-view method DiNMF

    example:
    V_star, obj_val, cnt = DiNMF(n_components=class_num, alpha=0.001, beta=0.001).fit(X)
    X[0]: m0*n
    X[1]: m1*n
    ...
    V_star: n_components*n
    """

    def __init__(self, n_components, alpha, beta,
                 max_iter=1000, tol=1e-5, kmeans_init=True) -> None:
        self.n_components = n_components
        self.kmeans_init = kmeans_init
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.beta = beta

    def get_info(self, X):
        view_num = len(X)
        m = [X[i].shape[0] for i in range(view_num)]
        n = X[0].shape[1]
        return view_num, m, n

    def init_W_H(self, X):
        view_num, m, n = self.get_info(X)
        n_components = self.n_components

        W = np.empty(view_num, dtype=object)
        H = np.empty(view_num, dtype=object)

        if self.kmeans_init:
            # Use kmeans for initialization
            for v in range(view_num):
                W[v], H[v] = init_by_kmeans(X[v], n_components)
        else:
            # random
            for v in range(view_num):
                W[v], H[v] = random_init(X[v], n_components)
        return W, H

    def sum_Hv(self, H, v, view_num):
        res = np.zeros_like(H[0])
        for w in range(view_num):
            if w != v:
                res += H[w]
        return res

    def sum_trace_HH(self, H, v, view_num):
        res = 0
        for w in range(view_num):
            if w != v:
                res += np.trace(H[v] @ H[w].T)
        return res

    def calc_obj(self, X, W, H, view_num):
        alpha = self.alpha
        beta = self.beta
        obj = 0
        for v in range(view_num):
            obj += ((np.linalg.norm(X[v] - W[v] @ H[v], ord='fro')) ** 2) \
                   + alpha * self.sum_trace_HH(H, v, view_num) \
                   + beta * (np.linalg.norm(H[v], ord='fro') ** 2)
        return obj

    def fit(self, X):
        X = X.copy()
        max_iter = self.max_iter
        alpha = self.alpha
        beta = self.beta
        tol = self.tol
        view_num, m, n = self.get_info(X)
        # init
        for v in range(view_num):
            X[v] = normalize(X[v].T, norm='l2').T
            # X[v] = l2_normlize_data(X[v])
        W, H = self.init_W_H(X)

        obj = []
        error_cnt = 0
        # update
        for t in range(max_iter):
            for v in range(view_num):
                # update H[v]
                tempUp = 2 * W[v].T @ X[v]
                tempDown = 2 * W[v].T @ W[v] @ H[v] + alpha * self.sum_Hv(H, v, view_num) + 2 * beta * H[v]
                H[v] = H[v] * (tempUp / np.maximum(tempDown, 1e-9))

                # update W[v]
                tempUp = X[v] @ H[v].T
                tempDown = W[v] @ H[v] @ H[v].T
                W[v] = W[v] * (tempUp / np.maximum(tempDown, 1e-9))

            obj.append(self.calc_obj(X, W, H, view_num))
            print('iter = {}, obj = {:.2f}'.format(t + 1, obj[t]))

            if t > 0 and obj[t] > obj[t - 1]:
                error_cnt += 1
            if t > 0 and np.abs(obj[t] - obj[t - 1]) / obj[t - 1] <= tol:
                break

        H_star = np.zeros_like(H[0])
        for v in range(view_num):
            H_star += 1 / view_num * H[v]
        return H_star, obj, error_cnt
