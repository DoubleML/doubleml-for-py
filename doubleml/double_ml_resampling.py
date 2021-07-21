import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import KFold, RepeatedKFold
import itertools


class DoubleMLResampling:
    def __init__(self,
                 n_folds,
                 n_rep,
                 n_obs,
                 apply_cross_fitting):
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.n_obs = n_obs
        self.apply_cross_fitting = apply_cross_fitting
        if (self.n_folds == 1) & self.apply_cross_fitting:
            warnings.warn('apply_cross_fitting is set to False. Cross-fitting is not supported for n_folds = 1.')
            self.apply_cross_fitting = False
        if not apply_cross_fitting:
            assert n_folds <= 2
        self.resampling = RepeatedKFold(n_splits=n_folds,
                                        n_repeats=n_rep)

        if n_folds == 1:
            assert n_rep == 1
            self.resampling = ResampleNoSplit()

    def split_samples(self):
        all_smpls = [(train, test) for train, test in self.resampling.split(np.zeros(self.n_obs))]
        smpls = [all_smpls[(i_repeat * self.n_folds):((i_repeat + 1) * self.n_folds)]
                 for i_repeat in range(self.n_rep)]
        if not self.apply_cross_fitting:
            # in the no cross-fitting case in each repetition we only use the first sample split
            smpls = [[xx[0]] for xx in smpls]
        return smpls


# A helper class to run double without cross-fitting
class ResampleNoSplit():
    def __init__(self):
        self.n_splits = 1

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        indices = np.arange(X.shape[0])
        yield indices, indices


class DoubleMLClusterResampling:
    def __init__(self,
                 n_folds,
                 n_rep,
                 n_obs,
                 apply_cross_fitting,
                 n_cluster_vars,
                 cluster_vars):
        if (n_folds == 1) | (not apply_cross_fitting) | (n_rep > 1):
            raise NotImplementedError('Repeated cross-fitting (`n_rep > 1`) '
                                      'and no cross-fitting (`apply_cross_fitting = False`) '
                                      'are not yet implemented with clustering.')
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.n_obs = n_obs
        self.apply_cross_fitting = apply_cross_fitting

        assert cluster_vars.shape[0] == n_obs
        assert cluster_vars.shape[1] == n_cluster_vars
        self.n_cluster_vars = n_cluster_vars
        self.cluster_vars = cluster_vars
        self.resampling = KFold(n_splits=n_folds, shuffle=True)

    def split_samples(self):
        smpls_cluster_vars = []
        for i_var in range(self.n_cluster_vars):
            this_cluster_var = self.cluster_vars[:, i_var]
            clusters = np.unique(this_cluster_var)
            n_clusters = len(clusters)
            smpls_cluster_vars.append([(clusters[train], clusters[test])
                                       for train, test in self.resampling.split(np.zeros(n_clusters))])

        smpls = []
        smpls_cluster = []
        # build the cartesian product
        cart = np.array(np.meshgrid(*[np.arange(self.n_folds)
                                      for i in range(self.n_cluster_vars)])).T.reshape(-1, self.n_cluster_vars)
        for i_smpl in range(cart.shape[0]):
            ind_train = np.full(self.n_obs, True)
            ind_test = np.full(self.n_obs, True)
            this_cluster_smpl_train = []
            this_cluster_smpl_test = []
            for i_var in range(self.n_cluster_vars):
                i_fold = cart[i_smpl, i_var]
                train_clusters = smpls_cluster_vars[i_var][i_fold][0]
                test_clusters = smpls_cluster_vars[i_var][i_fold][1]
                this_cluster_smpl_train.append(train_clusters)
                this_cluster_smpl_test.append(test_clusters)
                ind_train = ind_train & np.in1d(self.cluster_vars[:, i_var], train_clusters)
                ind_test = ind_test & np.in1d(self.cluster_vars[:, i_var], test_clusters)
            train_set = np.arange(self.n_obs)[ind_train]
            test_set = np.arange(self.n_obs)[ind_test]
            smpls.append((train_set, test_set))
            smpls_cluster.append((this_cluster_smpl_train, this_cluster_smpl_test))

        return [smpls], [smpls_cluster]


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
