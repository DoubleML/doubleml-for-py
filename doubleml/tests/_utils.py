import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.base import clone
import pandas as pd
from scipy.stats import norm

from ..utils._estimation import _var_est, _aggregate_coefs_and_ses
from ..double_ml_data import DoubleMLBaseData


class DummyDataClass(DoubleMLBaseData):
    def __init__(self,
                 data):
        DoubleMLBaseData.__init__(self, data)

    @property
    def n_coefs(self):
        return 1


def draw_smpls(n_obs, n_folds, n_rep=1, groups=None):
    all_smpls = []
    for _ in range(n_rep):
        if groups is None:
            resampling = KFold(n_splits=n_folds,
                               shuffle=True)
        else:
            resampling = StratifiedKFold(n_splits=n_folds,
                                         shuffle=True)
        smpls = [(train, test) for train, test in resampling.split(X=np.zeros(n_obs), y=groups)]
        all_smpls.append(smpls)
    return all_smpls


def fit_predict(y, x, ml_model, params, smpls, train_cond=None):
    y_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if params is not None:
            ml_model.set_params(**params[idx])
        if train_cond is None:
            y_hat.append(ml_model.fit(x[train_index], y[train_index]).predict(x[test_index]))
        else:
            train_index_cond = np.intersect1d(train_cond, train_index)
            y_hat.append(ml_model.fit(x[train_index_cond], y[train_index_cond]).predict(x[test_index]))

    return y_hat


def fit_predict_proba(y, x, ml_model, params, smpls, trimming_threshold=0, train_cond=None):
    y_hat = []
    for idx, (train_index, test_index) in enumerate(smpls):
        if params is not None:
            ml_model.set_params(**params[idx])
        if train_cond is None:
            preds = ml_model.fit(x[train_index], y[train_index]).predict_proba(x[test_index])[:, 1]
        else:
            train_index_cond = np.intersect1d(train_cond, train_index)
            preds = ml_model.fit(x[train_index_cond], y[train_index_cond]).predict_proba(x[test_index])[:, 1]

        if trimming_threshold > 0:
            preds[preds < trimming_threshold] = trimming_threshold
            preds[preds > 1 - trimming_threshold] = 1 - trimming_threshold
        y_hat.append(preds)

    return y_hat


def tune_grid_search(y, x, ml_model, smpls, param_grid, n_folds_tune, train_cond=None):
    tune_res = [None] * len(smpls)
    for idx, (train_index, _) in enumerate(smpls):
        g_tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        g_grid_search = GridSearchCV(ml_model, param_grid,
                                     cv=g_tune_resampling)
        if train_cond is None:
            tune_res[idx] = g_grid_search.fit(x[train_index, :], y[train_index])
        else:
            train_index_cond = np.intersect1d(train_cond, train_index)
            tune_res[idx] = g_grid_search.fit(x[train_index_cond, :], y[train_index_cond])

    return tune_res


def _clone(learner):
    if learner is None:
        res = None
    else:
        res = clone(learner)
    return res


def generate_dml_dict(psi_a, psi_b):
    n_obs = psi_a.shape[0]
    n_thetas = psi_a.shape[1]
    n_rep = psi_a.shape[2]

    all_thetas = -1.0*np.mean(psi_b, axis=0)
    all_ses = np.zeros(shape=(n_thetas, n_rep))
    for i_rep in range(n_rep):
        for i_theta in range(n_thetas):
            psi = psi_a[:, i_theta, i_rep]*all_thetas[i_theta, i_rep] + psi_b[:, i_theta, i_rep]
            var_estimate, _ = _var_est(
                psi=psi,
                psi_deriv=psi_a[:, i_theta, i_rep],
                smpls=None,
                is_cluster_data=False
            )
            all_ses[i_theta, i_rep] = np.sqrt(var_estimate)

    var_scaling_factors = np.full(n_thetas, n_obs)
    thetas, ses = _aggregate_coefs_and_ses(
        all_coefs=all_thetas,
        all_ses=all_ses,
        var_scaling_factors=var_scaling_factors,
    )
    scaled_psi = psi_b / np.mean(psi_a, axis=0)

    doubleml_dict = {
        'thetas': thetas,
        'ses': ses,
        'all_thetas': all_thetas,
        'all_ses': all_ses,
        'var_scaling_factors': var_scaling_factors,
        'scaled_psi': scaled_psi,
    }

    return doubleml_dict


def confint_manual(coef, se, index_names, boot_t_stat=None, joint=True, level=0.95):
    a = (1 - level)
    ab = np.array([a / 2, 1. - a / 2])
    if joint:
        assert boot_t_stat.shape[2] == 1
        sim = np.amax(np.abs(boot_t_stat[:, :, 0]), 1)
        hatc = np.quantile(sim, 1 - a)
        ci = np.vstack((coef - se * hatc, coef + se * hatc)).T
    else:
        fac = norm.ppf(ab)
        ci = np.vstack((coef + se * fac[0], coef + se * fac[1])).T

    df_ci = pd.DataFrame(ci,
                         columns=['{:.1f} %'.format(i * 100) for i in ab],
                         index=index_names)
    return df_ci
