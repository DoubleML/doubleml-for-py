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


def est_one_way_cluster_dml2(psi_a, psi_b, cluster_var, smpls_one_split):
    psi_a_subsample = 0.
    psi_b_subsample = 0.
    for (_, test_index) in smpls_one_split:
        I_k = np.unique(cluster_var[test_index])
        const = 1/len(I_k)
        psi_a_subsample += const*np.sum(psi_a[test_index])
        psi_b_subsample += const*np.sum(psi_b[test_index])
    theta = -psi_b_subsample / psi_a_subsample
    return theta


def var_one_way_cluster(psi, psi_a, cluster_var, smpls_one_split):
    gamma_hat = 0
    j_hat = 0
    for (_, test_index) in smpls_one_split:
        I_k = np.unique(cluster_var[test_index])
        const = 1/len(I_k)
        for i in I_k:
            ind = cluster_var == i
            for val_i in psi[ind]:
                for val_j in psi[ind]:
                    gamma_hat += const * val_i * val_j
        j_hat += np.sum(psi_a[test_index])/len(I_k)
    n_folds = len(smpls_one_split)
    gamma_hat = gamma_hat/n_folds
    j_hat = j_hat/n_folds
    var = gamma_hat / (j_hat ** 2) / len(np.unique(cluster_var))
    return var


def est_two_way_cluster_dml2(psi_a, psi_b, cluster_var1, cluster_var2, smpls_one_split):
    psi_a_subsample = 0.
    psi_b_subsample = 0.
    for (_, test_index) in smpls_one_split:
        I_k = np.unique(cluster_var1[test_index])
        J_l = np.unique(cluster_var2[test_index])
        const = 1/(len(I_k) * len(J_l))
        psi_a_subsample += const*np.sum(psi_a[test_index])
        psi_b_subsample += const*np.sum(psi_b[test_index])
    theta = -psi_b_subsample / psi_a_subsample
    return theta


def var_two_way_cluster(psi, psi_a, cluster_var1, cluster_var2, smpls_one_split):
    gamma_hat = 0
    j_hat = 0
    for (_, test_index) in smpls_one_split:
        I_k = np.unique(cluster_var1[test_index])
        J_l = np.unique(cluster_var2[test_index])
        const = min(len(I_k), len(J_l))/(len(I_k)*len(J_l))**2
        for i in I_k:
            for j in J_l:
                for j_ in J_l:
                    ind1 = (cluster_var1 == i) & (cluster_var2 == j)
                    ind2 = (cluster_var1 == i) & (cluster_var2 == j_)
                    gamma_hat += const * psi[ind1] * psi[ind2]
        for j in J_l:
            for i in I_k:
                for i_ in I_k:
                    ind1 = (cluster_var1 == i) & (cluster_var2 == j)
                    ind2 = (cluster_var1 == i_) & (cluster_var2 == j)
                    gamma_hat += const * psi[ind1] * psi[ind2]
        j_hat += np.sum(psi_a[test_index])/(len(I_k)*len(J_l))
    n_folds = len(smpls_one_split)
    gamma_hat = gamma_hat/n_folds
    j_hat = j_hat/n_folds
    n_clusters1 = len(np.unique(cluster_var1))
    n_clusters2 = len(np.unique(cluster_var2))
    var_scaling_factor = min(n_clusters1, n_clusters2)
    var = gamma_hat / (j_hat ** 2) / var_scaling_factor
    return var
