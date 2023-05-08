import numpy as np
import warnings
from scipy.optimize import minimize_scalar

from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.utils.multiclass import type_of_target

from statsmodels.nonparametric.kde import KDEUnivariate

from joblib import Parallel, delayed


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


def _check_is_partition(smpls, n_obs):
    test_indices = np.concatenate([test_index for _, test_index in smpls])
    if len(test_indices) != n_obs:
        return False
    hit = np.zeros(n_obs, dtype=bool)
    hit[test_indices] = True
    if not np.all(hit):
        return False
    return True


def _check_all_smpls(all_smpls, n_obs, check_intersect=False):
    all_smpls_checked = list()
    for smpl in all_smpls:
        all_smpls_checked.append(_check_smpl_split(smpl, n_obs, check_intersect))
    return all_smpls_checked


def _check_smpl_split(smpl, n_obs, check_intersect=False):
    smpl_checked = list()
    for tpl in smpl:
        smpl_checked.append(_check_smpl_split_tpl(tpl, n_obs, check_intersect))
    return smpl_checked


def _check_smpl_split_tpl(tpl, n_obs, check_intersect=False):
    train_index = np.sort(np.array(tpl[0]))
    test_index = np.sort(np.array(tpl[1]))

    if not issubclass(train_index.dtype.type, np.integer):
        raise TypeError('Invalid sample split. Train indices must be of type integer.')
    if not issubclass(test_index.dtype.type, np.integer):
        raise TypeError('Invalid sample split. Test indices must be of type integer.')

    if check_intersect:
        if set(train_index) & set(test_index):
            raise ValueError('Invalid sample split. Intersection of train and test indices is not empty.')

    if len(np.unique(train_index)) != len(train_index):
        raise ValueError('Invalid sample split. Train indices contain non-unique entries.')
    if len(np.unique(test_index)) != len(test_index):
        raise ValueError('Invalid sample split. Test indices contain non-unique entries.')

    # we sort the indices above
    # if not np.all(np.diff(train_index) > 0):
    #     raise NotImplementedError('Invalid sample split. Only sorted train indices are supported.')
    # if not np.all(np.diff(test_index) > 0):
    #     raise NotImplementedError('Invalid sample split. Only sorted test indices are supported.')

    if not set(train_index).issubset(range(n_obs)):
        raise ValueError('Invalid sample split. Train indices must be in [0, n_obs).')
    if not set(test_index).issubset(range(n_obs)):
        raise ValueError('Invalid sample split. Test indices must be in [0, n_obs).')

    return train_index, test_index


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
            if not np.alltrue(fold_ids == np.arange(len(smpls))):
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


def _check_finite_predictions(preds, learner, learner_name, smpls):
    test_indices = np.concatenate([test_index for _, test_index in smpls])
    if not np.all(np.isfinite(preds[test_indices])):
        raise ValueError(f'Predictions from learner {str(learner)} for {learner_name} are not finite.')
    return


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


def _check_is_propensity(preds, learner, learner_name, smpls, eps=1e-12):
    test_indices = np.concatenate([test_index for _, test_index in smpls])
    if any((preds[test_indices] < eps) | (preds[test_indices] > 1 - eps)):
        warnings.warn(f'Propensity predictions from learner {str(learner)} for'
                      f' {learner_name} are close to zero or one (eps={eps}).')
    return


def _rmse(y_true, y_pred):
    subset = np.logical_not(np.isnan(y_true))
    rmse = mean_squared_error(y_true[subset], y_pred[subset], squared=False)
    return rmse


def _predict_zero_one_propensity(learner, X):
    pred_proba = learner.predict_proba(X)
    if pred_proba.shape[1] == 2:
        res = pred_proba[:, 1]
    else:
        warnings.warn("Subsample has not common support. Results are based on adjusted propensities.")
        res = learner.predict(X)
    return res


def _check_contains_iv(obj_dml_data):
    if obj_dml_data.z_cols is not None:
        raise ValueError('Incompatible data. ' +
                         ' and '.join(obj_dml_data.z_cols) +
                         ' have been set as instrumental variable(s). '
                         'To fit an local model see the documentation.')


def _check_zero_one_treatment(obj_dml):
    one_treat = (obj_dml._dml_data.n_treat == 1)
    binary_treat = (type_of_target(obj_dml._dml_data.d) == 'binary')
    zero_one_treat = np.all((np.power(obj_dml._dml_data.d, 2) - obj_dml._dml_data.d) == 0)
    if not (one_treat & binary_treat & zero_one_treat):
        raise ValueError('Incompatible data. '
                         f'To fit an {str(obj_dml.score)} model with DML '
                         'exactly one binary variable with values 0 and 1 '
                         'needs to be specified as treatment variable.')


def _check_quantile(quantile):
    if not isinstance(quantile, float):
        raise TypeError('Quantile has to be a float. ' +
                        f'Object of type {str(type(quantile))} passed.')

    if (quantile <= 0) | (quantile >= 1):
        raise ValueError('Quantile has be between 0 or 1. ' +
                         f'Quantile {str(quantile)} passed.')
    return


def _check_treatment(treatment):
    if not isinstance(treatment, int):
        raise TypeError('Treatment indicator has to be an integer. ' +
                        f'Object of type {str(type(treatment))} passed.')

    if (treatment != 0) & (treatment != 1):
        raise ValueError('Treatment indicator has be either 0 or 1. ' +
                         f'Treatment indicator {str(treatment)} passed.')
    return


def _check_trimming(trimming_rule, trimming_threshold):
    valid_trimming_rule = ['truncate']
    if trimming_rule not in valid_trimming_rule:
        raise ValueError('Invalid trimming_rule ' + str(trimming_rule) + '. ' +
                         'Valid trimming_rule ' + ' or '.join(valid_trimming_rule) + '.')
    if not isinstance(trimming_threshold, float):
        raise TypeError('trimming_threshold has to be a float. ' +
                        f'Object of type {str(type(trimming_threshold))} passed.')
    if (trimming_threshold <= 0) | (trimming_threshold >= 0.5):
        raise ValueError('Invalid trimming_threshold ' + str(trimming_threshold) + '. ' +
                         'trimming_threshold has to be between 0 and 0.5.')
    return


def _check_score(score, valid_score, allow_callable=True):
    if isinstance(score, str):
        if score not in valid_score:
            raise ValueError('Invalid score ' + score + '. ' +
                             'Valid score ' + ' or '.join(valid_score) + '.')
    else:
        if allow_callable:
            if not callable(score):
                raise TypeError('score should be either a string or a callable. '
                                '%r was passed.' % score)
        else:
            raise TypeError('score should be a string. '
                            '%r was passed.' % score)
    return


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
