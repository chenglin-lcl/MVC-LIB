import numpy as np
import math
from scipy.optimize import linear_sum_assignment as linear_assignment


class Metric(object):
    def __init__(self, true, pred) -> None:
        self.true = true
        self.pred = pred

    def calc_ACC(self):
        """
        Calculate clustering accuracy. Require scipy installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = self.true
        y_pred = self.pred
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size


    def calc_NMI(self):
        A = self.true
        B = self.pred
        total = len(A)
        A_ids = set(A)
        B_ids = set(B)
        # MI
        MI = 0
        eps = 1.4e-45
        for idA in A_ids:
            for idB in B_ids:
                idAOccur = np.where(A==idA)
                idBOccur = np.where(B==idB)
                idABOccur = np.intersect1d(idAOccur,idBOccur)
                px = 1.0*len(idAOccur[0])/total
                py = 1.0*len(idBOccur[0])/total
                pxy = 1.0*len(idABOccur)/total
                MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
        # NMI
        Hx = 0
        for idA in A_ids:
            idAOccurCount = 1.0*len(np.where(A==idA)[0])
            Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
        Hy = 0
        for idB in B_ids:
            idBOccurCount = 1.0*len(np.where(B==idB)[0])
            Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
        # MIhat = 2.0*MI/(Hx+Hy)
        MIhat = MI / max(Hx, Hy)
        return MIhat

    def calc_Purity(self):
        label = self.true
        cluster = self.pred
        cluster = np.array(cluster)
        label = np. array(label)
        indedata1 = {}
        for p in np.unique(label):
            indedata1[p] = np.argwhere(label == p)
        indedata2 = {}
        for q in np.unique(cluster):
            indedata2[q] = np.argwhere(cluster == q)

        count_all = []
        for i in indedata1.values():
            count = []
            for j in indedata2.values():
                a = np.intersect1d(i, j).shape[0]
                count.append(a)
            count_all.append(count)

        return sum(np.max(count_all, axis=0))/len(cluster)
