import numpy as np
import warnings
from scipy.optimize import minimize_scalar

from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error

from statsmodels.nonparametric.kde import KDEUnivariate

from joblib import Parallel, delayed

from ._checks import _check_is_partition


def _assure_2d_array(x):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim > 2:
        raise ValueError('Only one- or two-dimensional arrays are allowed')
    return x


def _get_cond_smpls(smpls, bin_var):
    smpls_0 = [(np.intersect1d(np.where(bin_var == 0)[0], train), test) for train, test in smpls]
    smpls_1 = [(np.intersect1d(np.where(bin_var == 1)[0], train), test) for train, test in smpls]
    return smpls_0, smpls_1


def _get_cond_smpls_2d(smpls, bin_var1, bin_var2):
    subset_00 = (bin_var1 == 0) & (bin_var2 == 0)
    smpls_00 = [(np.intersect1d(np.where(subset_00)[0], train), test) for train, test in smpls]
    subset_01 = (bin_var1 == 0) & (bin_var2 == 1)
    smpls_01 = [(np.intersect1d(np.where(subset_01)[0], train), test) for train, test in smpls]
    subset_10 = (bin_var1 == 1) & (bin_var2 == 0)
    smpls_10 = [(np.intersect1d(np.where(subset_10)[0], train), test) for train, test in smpls]
    subset_11 = (bin_var1 == 1) & (bin_var2 == 1)
    smpls_11 = [(np.intersect1d(np.where(subset_11)[0], train), test) for train, test in smpls]
    return smpls_00, smpls_01, smpls_10, smpls_11


def _fit(estimator, x, y, train_index, idx=None):
    estimator.fit(x[train_index, :], y[train_index])
    return estimator, idx


def _dml_cv_predict(estimator, x, y, smpls=None,
                    n_jobs=None, est_params=None, method='predict', return_train_preds=False, return_models=False):
    n_obs = x.shape[0]

    smpls_is_partition = _check_is_partition(smpls, n_obs)
    fold_specific_params = (est_params is not None) & (not isinstance(est_params, dict))
    fold_specific_target = isinstance(y, list)
    manual_cv_predict = (not smpls_is_partition) | return_train_preds | fold_specific_params | fold_specific_target \
        | return_models

    res = {'models': None}
    if not manual_cv_predict:
        if est_params is None:
            # if there are no parameters set we redirect to the standard method
            preds = cross_val_predict(clone(estimator), x, y, cv=smpls, n_jobs=n_jobs, method=method)
        else:
            assert isinstance(est_params, dict)
            # if no fold-specific parameters we redirect to the standard method
            # warnings.warn("Using the same (hyper-)parameters for all folds")
            preds = cross_val_predict(clone(estimator).set_params(**est_params), x, y, cv=smpls, n_jobs=n_jobs,
                                      method=method)
        if method == 'predict_proba':
            res['preds'] = preds[:, 1]
        else:
            res['preds'] = preds
        res['targets'] = np.copy(y)
    else:
        if not smpls_is_partition:
            assert not fold_specific_target, 'combination of fold-specific y and no cross-fitting not implemented yet'
            assert len(smpls) == 1

        if method == 'predict_proba':
            assert not fold_specific_target  # fold_specific_target only needed for PLIV.partialXZ
            y = np.asarray(y)
            le = LabelEncoder()
            y = le.fit_transform(y)

        parallel = Parallel(n_jobs=n_jobs, verbose=0, pre_dispatch='2*n_jobs')

        if fold_specific_target:
            y_list = list()
            for idx, (train_index, _) in enumerate(smpls):
                xx = np.full(n_obs, np.nan)
                xx[train_index] = y[idx]
                y_list.append(xx)
        else:
            # just replicate the y in a list
            y_list = [y] * len(smpls)

        if est_params is None:
            fitted_models = parallel(delayed(_fit)(
                clone(estimator), x, y_list[idx], train_index, idx)
                                     for idx, (train_index, test_index) in enumerate(smpls))
        elif isinstance(est_params, dict):
            # warnings.warn("Using the same (hyper-)parameters for all folds")
            fitted_models = parallel(delayed(_fit)(
                clone(estimator).set_params(**est_params), x, y_list[idx], train_index, idx)
                                     for idx, (train_index, test_index) in enumerate(smpls))
        else:
            assert len(est_params) == len(smpls), 'provide one parameter setting per fold'
            fitted_models = parallel(delayed(_fit)(
                clone(estimator).set_params(**est_params[idx]), x, y_list[idx], train_index, idx)
                                     for idx, (train_index, test_index) in enumerate(smpls))

        preds = np.full(n_obs, np.nan)
        targets = np.full(n_obs, np.nan)
        train_preds = list()
        train_targets = list()
        for idx, (train_index, test_index) in enumerate(smpls):
            assert idx == fitted_models[idx][1]
            pred_fun = getattr(fitted_models[idx][0], method)
            if method == 'predict_proba':
                preds[test_index] = pred_fun(x[test_index, :])[:, 1]
            else:
                preds[test_index] = pred_fun(x[test_index, :])

            if fold_specific_target:
                # targets not available for fold specific target
                targets = None
            else:
                targets[test_index] = y[test_index]

            if return_train_preds:
                train_preds.append(pred_fun(x[train_index, :]))
                train_targets.append(y[train_index])

        res['preds'] = preds
        res['targets'] = targets
        if return_train_preds:
            res['train_preds'] = train_preds
            res['train_targets'] = train_targets
        if return_models:
            fold_ids = [xx[1] for xx in fitted_models]
            if not np.all(fold_ids == np.arange(len(smpls))):
                raise RuntimeError('export of fitted models failed')
            res['models'] = [xx[0] for xx in fitted_models]

    return res


def _dml_tune(y, x, train_inds,
              learner, param_grid, scoring_method,
              n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search):
    tune_res = list()
    for train_index in train_inds:
        tune_resampling = KFold(n_splits=n_folds_tune, shuffle=True)
        if search_mode == 'grid_search':
            g_grid_search = GridSearchCV(learner, param_grid,
                                         scoring=scoring_method,
                                         cv=tune_resampling, n_jobs=n_jobs_cv)
        else:
            assert search_mode == 'randomized_search'
            g_grid_search = RandomizedSearchCV(learner, param_grid,
                                               scoring=scoring_method,
                                               cv=tune_resampling, n_jobs=n_jobs_cv,
                                               n_iter=n_iter_randomized_search)
        tune_res.append(g_grid_search.fit(x[train_index, :], y[train_index]))

    return tune_res


def _draw_weights(method, n_rep_boot, n_obs):
    if method == 'Bayes':
        weights = np.random.exponential(scale=1.0, size=(n_rep_boot, n_obs)) - 1.
    elif method == 'normal':
        weights = np.random.normal(loc=0.0, scale=1.0, size=(n_rep_boot, n_obs))
    elif method == 'wild':
        xx = np.random.normal(loc=0.0, scale=1.0, size=(n_rep_boot, n_obs))
        yy = np.random.normal(loc=0.0, scale=1.0, size=(n_rep_boot, n_obs))
        weights = xx / np.sqrt(2) + (np.power(yy, 2) - 1) / 2
    else:
        raise ValueError('invalid boot method')

    return weights


def _trimm(preds, trimming_rule, trimming_threshold):
    if trimming_rule == 'truncate':
        preds[preds < trimming_threshold] = trimming_threshold
        preds[preds > 1 - trimming_threshold] = 1 - trimming_threshold
    return preds


def _normalize_ipw(propensity, treatment):
    mean_treat1 = np.mean(np.divide(treatment, propensity))
    mean_treat0 = np.mean(np.divide(1.0-treatment, 1.0-propensity))
    normalized_weights = np.multiply(treatment, np.multiply(propensity, mean_treat1)) \
        + np.multiply(1.0-treatment, 1.0 - np.multiply(1.0-propensity, mean_treat0))

    return normalized_weights


def _rmse(y_true, y_pred):
    subset = np.logical_not(np.isnan(y_true))
    rmse = root_mean_squared_error(y_true[subset], y_pred[subset])
    return rmse


def _predict_zero_one_propensity(learner, X):
    pred_proba = learner.predict_proba(X)
    if pred_proba.shape[1] == 2:
        res = pred_proba[:, 1]
    else:
        warnings.warn("Subsample has not common support. Results are based on adjusted propensities.")
        res = learner.predict(X)
    return res


def _get_bracket_guess(score, coef_start, coef_bounds):
    max_bracket_length = coef_bounds[1] - coef_bounds[0]
    b_guess = coef_bounds
    delta = 0.1
    s_different = False
    while (not s_different) & (delta <= 1.0):
        a = np.maximum(coef_start - delta * max_bracket_length / 2, coef_bounds[0])
        b = np.minimum(coef_start + delta * max_bracket_length / 2, coef_bounds[1])
        b_guess = (a, b)
        f_a = score(b_guess[0])
        f_b = score(b_guess[1])
        s_different = (np.sign(f_a) != np.sign(f_b))
        delta += 0.1
    return s_different, b_guess


def _default_kde(u, weights):
    dens = KDEUnivariate(u)
    dens.fit(kernel='gau', bw='silverman', weights=weights, fft=False)

    return dens.evaluate(0)


def _solve_ipw_score(ipw_score, bracket_guess):
    def abs_ipw_score(theta):
        return abs(ipw_score(theta))

    res = minimize_scalar(abs_ipw_score,
                          bracket=bracket_guess,
                          method='brent')
    ipw_est = res.x
    return ipw_est


def _aggregate_coefs_and_ses(all_coefs, all_ses, var_scaling_factors):
    # aggregation is done over dimension 1, such that the coefs and ses have to be of shape (n_coefs, n_rep)
    coefs = np.median(all_coefs, 1)
    coefs_deviations = np.square(all_coefs - coefs.reshape(-1, 1))

    rescaled_variances = np.multiply(np.square(all_ses), var_scaling_factors.reshape(-1, 1))

    var = np.median(rescaled_variances + coefs_deviations, 1)
    ses = np.sqrt(np.divide(var, var_scaling_factors))

    return coefs, ses


def _var_est(psi, psi_deriv, smpls, is_cluster_data,
             cluster_vars=None, smpls_cluster=None, n_folds_per_cluster=None):

    if not is_cluster_data:
        # psi and psi_deriv should be of shape (n_obs, ...)
        var_scaling_factor = psi.shape[0]

        J = np.mean(psi_deriv)
        gamma_hat = np.mean(np.square(psi))

    else:
        assert cluster_vars is not None
        assert smpls_cluster is not None
        assert n_folds_per_cluster is not None
        n_folds = len(smpls)

        # one cluster
        if cluster_vars.shape[1] == 1:
            first_cluster_var = cluster_vars[:, 0]
            clusters = np.unique(first_cluster_var)
            gamma_hat = 0
            j_hat = 0
            for i_fold in range(n_folds):
                test_inds = smpls[i_fold][1]
                test_cluster_inds = smpls_cluster[i_fold][1]
                I_k = test_cluster_inds[0]
                const = 1 / len(I_k)
                for cluster_value in I_k:
                    ind_cluster = (first_cluster_var == cluster_value)
                    gamma_hat += const * np.sum(np.outer(psi[ind_cluster], psi[ind_cluster]))
                j_hat += np.sum(psi_deriv[test_inds]) / len(I_k)

            var_scaling_factor = len(clusters)
            J = np.divide(j_hat, n_folds_per_cluster)
            gamma_hat = np.divide(gamma_hat, n_folds_per_cluster)

        else:
            assert cluster_vars.shape[1] == 2
            first_cluster_var = cluster_vars[:, 0]
            second_cluster_var = cluster_vars[:, 1]
            gamma_hat = 0
            j_hat = 0
            for i_fold in range(n_folds):
                test_inds = smpls[i_fold][1]
                test_cluster_inds = smpls_cluster[i_fold][1]
                I_k = test_cluster_inds[0]
                J_l = test_cluster_inds[1]
                const = np.divide(min(len(I_k), len(J_l)), (np.square(len(I_k) * len(J_l))))
                for cluster_value in I_k:
                    ind_cluster = (first_cluster_var == cluster_value) & np.in1d(second_cluster_var, J_l)
                    gamma_hat += const * np.sum(np.outer(psi[ind_cluster], psi[ind_cluster]))
                for cluster_value in J_l:
                    ind_cluster = (second_cluster_var == cluster_value) & np.in1d(first_cluster_var, I_k)
                    gamma_hat += const * np.sum(np.outer(psi[ind_cluster], psi[ind_cluster]))
                j_hat += np.sum(psi_deriv[test_inds]) / (len(I_k) * len(J_l))

            n_first_clusters = len(np.unique(first_cluster_var))
            n_second_clusters = len(np.unique(second_cluster_var))
            var_scaling_factor = min(n_first_clusters, n_second_clusters)
            J = np.divide(j_hat, np.square(n_folds_per_cluster))
            gamma_hat = np.divide(gamma_hat, np.square(n_folds_per_cluster))

    scaling = np.divide(1.0, np.multiply(var_scaling_factor, np.square(J)))
    sigma2_hat = np.multiply(scaling, gamma_hat)

    return sigma2_hat, var_scaling_factor


def _cond_targets(target, cond_sample):
    cond_target = target.astype(float)
    cond_target[np.invert(cond_sample)] = np.nan
    return cond_target


def _set_external_predictions(external_predictions, learners, treatment, i_rep):
    ext_prediction_dict = {}
    for learner in learners:
        if external_predictions is None:
            ext_prediction_dict[learner] = None
        elif learner in external_predictions[treatment].keys():
            if isinstance(external_predictions[treatment][learner], np.ndarray):
                ext_prediction_dict[learner] = external_predictions[treatment][learner][:, i_rep]
            else:
                ext_prediction_dict[learner] = None
        else:
            ext_prediction_dict[learner] = None
    return ext_prediction_dict
