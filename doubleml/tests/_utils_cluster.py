import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
import itertools


class DoubleMLMultiwayResampling:
    def __init__(self,
                 n_folds,
                 smpl_sizes):
        self.n_folds = n_folds
        self.smpl_sizes = smpl_sizes
        assert len(smpl_sizes), 'For DoubleMLMultiwayResampling mmultiple sample sizes need to be provided'
        self.n_ways = len(smpl_sizes)
        self.resampling = KFold(n_splits=n_folds, shuffle=True)

    def split_samples(self):
        xx = [range(x) for x in self.smpl_sizes]
        ind = pd.MultiIndex.from_product(xx)
        lin_ind = range(len(ind))
        multi_to_lin_ind = pd.Series(lin_ind, index=ind)

        n_ways = len(self.smpl_sizes)
        smpls = []
        for i_way in range(n_ways):
            smpls.append([(train, test) for train, test in self.resampling.split(np.zeros(self.smpl_sizes[i_way]))])

        smpls_multi_ind = []
        xx = n_ways*[range(self.n_folds)]
        for ind_index_set in itertools.product(*xx):
            smpls_train_list = [smpls[i][ind_index_set[i]][0] for i in range(n_ways)]
            smpls_test_list = [smpls[i][ind_index_set[i]][1] for i in range(n_ways)]

            smpls_multi_ind.append((pd.MultiIndex.from_product(smpls_train_list).values,
                                    pd.MultiIndex.from_product(smpls_test_list).values))

        smpls_lin_ind = [(multi_to_lin_ind.loc[x[0]].values,
                          multi_to_lin_ind.loc[x[1]].values) for x in smpls_multi_ind]

        return smpls_multi_ind, smpls_lin_ind


def var_one_way_cluster(psi, psi_a, cluster_var, smpls_one_split):
    gamma_hat = 0
    j_hat = 0
    for (_, test_index) in smpls_one_split:
        clusters = np.unique(cluster_var[test_index])
        const = 1/len(clusters)
        for i in clusters:
            ind = cluster_var == i
            for val_i in psi[ind]:
                for val_j in psi[ind]:
                    gamma_hat += const * val_i * val_j
        j_hat += np.sum(psi_a[test_index])/len(clusters)
    n_folds = len(smpls_one_split)
    gamma_hat = gamma_hat/n_folds
    j_hat = j_hat/n_folds
    var = gamma_hat / (j_hat ** 2) / len(np.unique(cluster_var))
    return var
