import numpy as np


class Graph(object):
    """
    construct a KNN adjacency matrix or weight matrix

    X: m*n
    m: the number of features
    n: the number of samples

    example:
    # m = 2, n = 9
    X = np.array([[1, 2, 4, 7, 3, 4, 8, 3, 6], [3, 4, 2, 3, 9, 0, 3, 10, 1]], dtype=np.float32)
    weight_matrix = Graph(neighbor_num=3).get_weight_matrix(X, sigma=1)
    print(weight_matrix)

    adjacency_matrix = Graph(neighbor_num=3).get_adjacency_matrix(X)
    print(adjacency_matrix)

    """

    def __init__(self, neighbor_num=5):
        self.neighbor_num = neighbor_num

    def calc_distance(self, X):
        # don't use broadcast mechanism
        _, n = X.shape
        distance = np.zeros([n, n], dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                dis = np.sqrt(np.sum((X[:, i] - X[:, j]) ** 2))
                distance[i][j] = dis
                distance[j][i] = dis
        return distance

    def get_am(self, X):
        """
        W_{ij} = 1 if x_{i} \in KNN{x_{j}} or x_{j} \in KNN{x_{i}}
               = 0 else
        return a binary adjacency matrix and distance
        """
        neighbor_num = self.neighbor_num
        _, n = X.shape
        # compute distance between different point
        distance = self.calc_distance(X)
        # Sort in ascending order
        idx = np.argsort(distance)
        a_mat = np.zeros(distance.shape, dtype=np.float32)
        # Select the first neighbor_num+1 samples
        select_idx = idx[:, :neighbor_num + 1]

        temp = np.zeros(select_idx.shape, dtype=np.int32)
        temp += np.linspace(0, n - 1, n, dtype=np.int32)[:, None]

        # location
        x_pos = temp.reshape(-1)
        y_pos = select_idx.reshape(-1)

        # write 
        # W_{ij} = 1 if x_{i} \in KNN{x_{j}} or x_{j} \in KNN{x_{i}}
        pos = tuple([x_pos, y_pos])
        a_mat[pos] = 1
        pos = tuple([y_pos, x_pos])
        a_mat[pos] = 1

        return a_mat, distance

    def get_adjacency_matrix(self, X):
        # binary adjacency matrix
        return self.get_am(X)[0]

    def get_weight_matrix(self, X, sigma=1):
        """
        W_{ij} = e^{||x_{i}-x_{j}||_{2}^{2}/(2*sigma^{2})} if x_{i} \in KNN{x_{j}} or x_{j} \in KNN{x_{i}}
               = 0                                          else
        return a weight matrix based on Heat kernel weight
        """
        b_mat, distance = self.get_am(X)
        HK_mat = np.exp(-distance * distance / (2 * sigma * sigma))
        HK_mat *= b_mat
        return HK_mat


class OptimalGraph(object):
    def __init__(self, neighbor_num=5):
        self.neighbor_num = neighbor_num

    def get_optimal_graph(self, X):
        view_num = len(X)
        res = Graph(self.neighbor_num).get_adjacency_matrix(X[0])
        for p in range(1, view_num):
            res = np.minimum(res, Graph(self.neighbor_num).get_adjacency_matrix(X[p]))
        return res
