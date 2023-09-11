import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from model.multi_view_methods import DiNMF
from utils import Metric
from data.dataloader import MVDatasets


def main(X, Y, class_num, alpha, beta):
    ACC = []
    NMI = []
    Purity = []
    err_cnt = []
    obj = []
    for t in range(10):
        V_star, obj_val, cnt = DiNMF(n_components=class_num, alpha=alpha, beta=beta).fit(X)
        y_pred = KMeans(n_clusters=class_num, n_init=10).fit_predict(V_star.T) + 1
        err_cnt.append(cnt)
        obj.append(obj_val)
        cluster_res = Metric(Y, y_pred)
        ACC.append(cluster_res.calc_ACC())
        NMI.append(cluster_res.calc_NMI())
        Purity.append(cluster_res.calc_Purity())
        print('ACC = {:.4f}, NMI = {:.4f}, Purity = {:.4f}'.format(ACC[t], NMI[t], Purity[t]))

    return ACC, NMI, Purity, obj[0], err_cnt[0]


if __name__ == '__main__':
    X, Y, class_num = MVDatasets().load_ORL_mtv()
    alpha = 0.1
    beta = 0.0001

    ACC, NMI, Purity, obj, err_cnt = main(X, Y, class_num, alpha=alpha, beta=beta)
    acc_mean = np.mean(ACC)
    acc_std = np.std(ACC)
    nmi_mean = np.mean(NMI)
    nmi_std = np.std(NMI)
    purity_mean = np.mean(Purity)
    purity_std = np.std(Purity)

    file_name = './results/ORL_results.txt'
    with open(file_name, 'a') as f:
        f.write('alpha = {}, beta = {}, err_cnt = {}:\n'.format( alpha, beta, err_cnt))
        f.write('\tACC = {:.4f} + {:.4f}, NMI = {:.4f} + {:.4f}, Purity = {:.4f} + {:.4f}\n'.format(
            acc_mean, acc_std, nmi_mean, nmi_std, purity_mean, purity_std))
    plt.plot(obj)
    plt.show()

