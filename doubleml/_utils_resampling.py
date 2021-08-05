import numpy as np
import warnings

from sklearn.model_selection import KFold, RepeatedKFold


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
        if (n_folds == 1) | (not apply_cross_fitting):
            raise NotImplementedError('No cross-fitting (`apply_cross_fitting = False`) '
                                      'is not yet implemented with clustering.')
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
        all_smpls = []
        all_smpls_cluster = []
        for _ in range(self.n_rep):
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
            all_smpls.append(smpls)
            all_smpls_cluster.append(smpls_cluster)

        return all_smpls, all_smpls_cluster
