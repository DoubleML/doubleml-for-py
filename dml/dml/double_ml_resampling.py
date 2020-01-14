import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold
import itertools



class DoubleMLResampling:
    def __init__(self,
                 n_folds,
                 n_rep_cross_fit,
                 n_obs):
        self.n_folds = n_folds
        self.n_rep_cross_fit = n_rep_cross_fit
        self.n_obs = n_obs
        self.resampling = RepeatedKFold(n_splits=n_folds,
                                        n_repeats=n_rep_cross_fit)

    def split_samples(self):
        all_smpls = [(train, test) for train, test in self.resampling.split(np.zeros(self.n_obs))]
        smpls = [all_smpls[(i_repeat * self.n_folds):((i_repeat + 1) * self.n_folds)]
                 for i_repeat in range(self.n_rep_cross_fit)]
        return smpls


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