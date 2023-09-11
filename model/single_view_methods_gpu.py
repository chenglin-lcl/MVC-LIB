from utils import init_by_kmeans, random_init
import torch
from sklearn.preprocessing import normalize
from utils import Graph


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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def trans_tensor(self, X):
        return torch.as_tensor(X).to(self.device)

    def init_W_H(self, X):
        n_components = self.n_components
        if self.kmeans_init:
            # Use kmeans for initialization
            W, H = init_by_kmeans(X, n_components)
        else:
            # random
            W, H = random_init(X, n_components)
        return W, H

    def calc_obj(self, X, W, H, L):
        obj = ((torch.linalg.norm(X - W @ H, ord='fro')) ** 2) + self.alpha * torch.trace(H @ L @ H.T)
        return obj

    def fit(self, X):
        X = X.copy()
        max_iter = self.max_iter
        tol = self.tol
        alpha = self.alpha
        obj = []
        error_cnt = 0
        # init WH on cpu
        W, H = self.init_W_H(X)
        # init graph on cpu
        S = Graph(neighbor_num=5).get_adjacency_matrix(X)
        # load on gpu
        X = self.trans_tensor(X)
        W = self.trans_tensor(W)
        H = self.trans_tensor(H)
        S = self.trans_tensor(S)
        D = torch.diag(torch.sum(S, dim=0))
        L = D - S

        # update on gpu
        for t in range(max_iter):

            # update H
            tempUp = W.T @ X + alpha * H @ S
            tempDown = W.T @ W @ H + alpha * H @ D
            H = H * (tempUp / tempDown.clamp_min(1e-9))

            # update W
            tempUp = X @ H.T
            tempDown = W @ H @ H.T
            W = W * (tempUp / tempDown.clamp_min(1e-9))

            obj.append(self.calc_obj(X, W, H, L).cpu())
            print('iter = {}, obj = {:.2f}'.format(t + 1, obj[t]))

            if t > 0 and obj[t] > obj[t - 1]:
                error_cnt += 1
            if t > 0 and torch.abs(obj[t] - obj[t - 1]) / torch.abs(obj[t - 1]) <= tol:
                break

        return H.cpu(), obj, error_cnt
