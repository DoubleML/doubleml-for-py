import numpy as np

from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold


class DoubleMLResampling:
    def __init__(self,
                 n_folds,
                 n_rep,
                 n_obs,
                 stratify=None):
        self.n_folds = n_folds
        self.n_rep = n_rep
        self.n_obs = n_obs
        self.stratify = stratify

        if n_folds < 2:
            raise ValueError('n_folds must be greater than 1. '
                             'You can use set_sample_splitting with a tuple to only use one fold.')

        if self.stratify is None:
            self.resampling = RepeatedKFold(n_splits=n_folds, n_repeats=n_rep)
        else:
            self.resampling = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_rep)

    def split_samples(self):
        all_smpls = [(train, test) for train, test in self.resampling.split(X=np.zeros(self.n_obs), y=self.stratify)]
        smpls = [all_smpls[(i_repeat * self.n_folds):((i_repeat + 1) * self.n_folds)]
                 for i_repeat in range(self.n_rep)]
        return smpls


class DoubleMLClusterResampling:
    def __init__(self,
                 n_folds,
                 n_rep,
                 n_obs,
                 n_cluster_vars,
                 cluster_vars):

        self.n_folds = n_folds
        self.n_rep = n_rep
        self.n_obs = n_obs

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
