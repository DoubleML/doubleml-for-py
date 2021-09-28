import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

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
                    n_jobs=None, est_params=None, method='predict', return_train_preds=False):
    n_obs = x.shape[0]

    smpls_is_partition = _check_is_partition(smpls, n_obs)
    fold_specific_params = (est_params is not None) & (not isinstance(est_params, dict))
    fold_specific_target = isinstance(y, list)
    manual_cv_predict = (not smpls_is_partition) | return_train_preds | fold_specific_params | fold_specific_target

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
            return preds[:, 1]
        else:
            return preds
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
        train_preds = list()
        for idx, (train_index, test_index) in enumerate(smpls):
            assert idx == fitted_models[idx][1]
            pred_fun = getattr(fitted_models[idx][0], method)
            if method == 'predict_proba':
                preds[test_index] = pred_fun(x[test_index, :])[:, 1]
            else:
                preds[test_index] = pred_fun(x[test_index, :])

            if return_train_preds:
                train_preds.append(pred_fun(x[train_index, :]))

        if return_train_preds:
            return preds, train_preds
        else:
            return preds


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
