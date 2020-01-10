import numpy as np
from sklearn.model_selection import RepeatedKFold


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

