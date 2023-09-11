from mat4py import loadmat
import numpy as np
import os


class MVDatasets(object):
    """
    load data for multi-view methods
    example:
    X, Y, class_num = MVDatasets().load_ORL_mtv()
    X[0]: m0*n
    X[1]: m1*n
    ...
    Y: n*1
    """

    def __init__(self, root='./data') -> None:
        self.root = root

    def load_ORL_mtv(self, out_view=3):
        root = self.root
        file_name = 'ORL_mtv.mat'
        view_num = 3
        file_dir = os.path.join(root, file_name)

        data = loadmat(file_dir)
        # print(list(data.keys()))

        X_data = np.empty(view_num, dtype=object)
        for view_idx in range(view_num):
            # [m, n]
            X_data[view_idx] = np.squeeze(np.array(data['X'][view_idx], dtype=np.float32)).T
            # print('feature dimension of ' + str(view_idx+1) + '-th view =', X_data[view_idx].shape[0])
        # [n,]
        Y = np.squeeze(np.array(data['Y'], dtype=np.float32)).T  # n*1
        if np.min(Y) == 0:
            Y += 1
        # print('the number of samples = ', Y.shape)

        class_num = len(set(Y))
        # print('class_num = ', class_num)
        return X_data[:out_view], Y, class_num

    def load_COIL20_mtv(self, out_view=3):
        root = self.root
        file_name = 'COIL20_mtv.mat'
        view_num = 3
        file_dir = os.path.join(root, file_name)

        data = loadmat(file_dir)
        # print(list(data.keys()))

        X_data = np.empty(view_num, dtype=object)
        for view_idx in range(view_num):
            # [m, n]
            X_data[view_idx] = np.squeeze(np.array(data['X'][view_idx], dtype=np.float32)).T
            # print('feature dimension of ' + str(view_idx+1) + '-th view =', X_data[view_idx].shape[0])
        # [n,]
        Y = np.squeeze(np.array(data['Y'], dtype=np.float32)).T  # n*1
        if np.min(Y) == 0:
            Y += 1
        # print('the number of samples = ', Y.shape)

        class_num = len(set(Y))
        # print('class_num = ', class_num)
        return X_data[:out_view], Y, class_num

    def load_cornell(self, out_view=2):
        root = self.root
        file_name = 'cornell.mat'
        file_dir = os.path.join(root, file_name)
        view_num = 2
        data = loadmat(file_dir)
        # print(list(data.keys()))

        X_data = np.empty(view_num, dtype=object)
        # [m, n]
        X_data[0] = np.squeeze(np.array(data['X1'], dtype=np.float32))
        X_data[1] = np.squeeze(np.array(data['X2'], dtype=np.float32))
        # print(X_data[0].shape)
        # print(X_data[1].shape)
        # [n,]
        Y = np.squeeze(np.array(data['Y'], dtype=np.float32)).T  # n*1
        if np.min(Y) == 0:
            Y += 1
        # print('the number of samples = ', Y.shape)

        class_num = len(set(Y))
        # print('class_num = ', class_num)
        return X_data[:out_view], Y, class_num

    def load_texas(self, out_view=2):
        root = self.root
        file_name = 'texas.mat'
        file_dir = os.path.join(root, file_name)
        view_num = 2
        data = loadmat(file_dir)
        print(list(data.keys()))

        X_data = np.empty(view_num, dtype=object)
        # [m, n]
        X_data[0] = np.squeeze(np.array(data['X1'], dtype=np.float32))
        X_data[1] = np.squeeze(np.array(data['X2'], dtype=np.float32))
        # [n,]
        Y = np.squeeze(np.array(data['Y'], dtype=np.float32)).T  # n*1
        if np.min(Y) == 0:
            Y += 1

        class_num = len(set(Y))

        return X_data[:out_view], Y, class_num

    def load_washington(self, out_view=2):
        root = self.root
        file_name = 'washington.mat'
        file_dir = os.path.join(root, file_name)
        view_num = 2
        data = loadmat(file_dir)
        print(list(data.keys()))

        X_data = np.empty(view_num, dtype=object)
        # [m, n]
        X_data[0] = np.squeeze(np.array(data['X1'], dtype=np.float32))
        X_data[1] = np.squeeze(np.array(data['X2'], dtype=np.float32))
        # [n,]
        Y = np.squeeze(np.array(data['Y'], dtype=np.float32)).T  # n*1
        if np.min(Y) == 0:
            Y += 1

        class_num = len(set(Y))

        return X_data[:out_view], Y, class_num

    def load_wisconsin(self, out_view=2):
        root = self.root
        file_name = 'wisconsin.mat'
        file_dir = os.path.join(root, file_name)
        view_num = 2
        data = loadmat(file_dir)
        print(list(data.keys()))

        X_data = np.empty(view_num, dtype=object)
        # [m, n]
        X_data[0] = np.squeeze(np.array(data['X1'], dtype=np.float32))
        X_data[1] = np.squeeze(np.array(data['X2'], dtype=np.float32))
        # [n,]
        Y = np.squeeze(np.array(data['Y'], dtype=np.float32)).T  # n*1
        if np.min(Y) == 0:
            Y += 1

        class_num = len(set(Y))

        return X_data[:out_view], Y, class_num

    def load_3Sources(self, out_view=3):
        root = self.root
        file_name = '3Sources.mat'
        file_dir = os.path.join(root, file_name)
        view_num = 3
        data = loadmat(file_dir)
        # print(list(data.keys()))
        # X = data['data'][0]

        X_data = np.empty(view_num, dtype=object)
        # [m, n]
        X_data[0] = np.squeeze(np.array(data['data'][0], dtype=np.float32))
        X_data[1] = np.squeeze(np.array(data['data'][1], dtype=np.float32))
        X_data[2] = np.squeeze(np.array(data['data'][2], dtype=np.float32))
        # [n,]
        Y = np.squeeze(np.array(data['truelabel'][0], dtype=np.float32)).T  # n*1
        if np.min(Y) == 0:
            Y += 1
        class_num = len(set(Y))

        return X_data[:out_view], Y, class_num

    def load_citeseer(self, out_view=2):
        root = self.root
        file_name = 'citeseer.mat'
        file_dir = os.path.join(root, file_name)
        view_num = 2
        data = loadmat(file_dir)
        # print(list(data.keys()))
        # X = data['X1']

        X_data = np.empty(view_num, dtype=object)
        # [m, n]
        X_data[0] = np.squeeze(np.array(data['X1'], dtype=np.float32)).T
        X_data[1] = np.squeeze(np.array(data['X2'], dtype=np.float32)).T
        # [n,]
        Y = np.squeeze(np.array(data['Y'], dtype=np.float32)).T  # n*1
        if np.min(Y) == 0:
            Y += 1
        class_num = len(set(Y))

        return X_data[:out_view], Y, class_num

    def load_reutersEN_0(self, out_view=3):
        root = self.root
        file_name = 'reutersEN_0.mat'
        file_dir = os.path.join(root, file_name)
        view_num = 5
        data = loadmat(file_dir)
        # print(list(data.keys()))
        # X = data['X1']

        X_data = np.empty(view_num, dtype=object)
        # [m, n]
        X_data[0] = np.squeeze(np.array(data['X1'], dtype=np.float32)).T
        X_data[1] = np.squeeze(np.array(data['X2'], dtype=np.float32)).T
        X_data[2] = np.squeeze(np.array(data['X3'], dtype=np.float32)).T
        X_data[3] = np.squeeze(np.array(data['X4'], dtype=np.float32)).T
        X_data[4] = np.squeeze(np.array(data['X5'], dtype=np.float32)).T
        # [n,]
        Y = np.squeeze(np.array(data['Y'], dtype=np.float32)).T  # n*1
        if np.min(Y) == 0:
            Y += 1
        class_num = len(set(Y))

        return X_data[:out_view], Y, class_num

