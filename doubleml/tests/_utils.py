import numpy as np
from sklearn.model_selection import KFold, GridSearchCV


def draw_smpls(n_obs, n_folds, n_rep=1):
    all_smpls = []
    for i_rep in range(n_rep):
        resampling = KFold(n_splits=n_folds,
                           shuffle=True)
        smpls = [(train, test) for train, test in resampling.split(np.zeros(n_obs))]
        all_smpls.append(smpls)
    return all_smpls
