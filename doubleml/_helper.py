import numpy as np

from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

import warnings
from joblib import Parallel, delayed


def assure_2d_array(x):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    elif x.ndim > 2:
        raise ValueError('Only one- or two-dimensional arrays are allowed')
    return x


def check_binary_vector(x, variable_name=''):
    # assure D binary
    assert type_of_target(x) == 'binary', 'variable ' + variable_name  + ' must be binary'
    
    if np.any(np.power(x, 2) - x != 0):
        raise ValueError('variable ' + variable_name + ' must be binary with values 0 and 1')


def _check_is_partition(smpls, n_obs):
    test_indices = np.concatenate([test_index for _, test_index in smpls])
    if len(test_indices) != n_obs:
        return False
    hit = np.zeros(n_obs, dtype=bool)
    hit[test_indices] = True
    if not np.all(hit):
        return False
    return True


def _fit(estimator, X, y, train_index, idx=None):
    estimator.fit(X[train_index, :], y[train_index])
    return estimator, idx


def _dml_cv_predict(estimator, X, y, smpls=None,
                    n_jobs=None, est_params=None, method='predict', return_train_preds=False):
    smpls_is_partition = _check_is_partition(smpls, len(y))
    fold_specific_params = (est_params is not None) & (not isinstance(est_params, dict))
    manual_cv_predict = (not smpls_is_partition) | (not return_train_preds) | fold_specific_params

    if not manual_cv_predict:
        if est_params is None:
            # if there are no parameters set we redirect to the standard method
            return cross_val_predict(estimator, X, y, cv=smpls, n_jobs=n_jobs, method=method)
        elif isinstance(est_params, dict):
            # if no fold-specific parameters we redirect to the standard method
            warnings.warn("Using the same (hyper-)parameters for all folds")
            return cross_val_predict(clone(estimator).set_params(**est_params), X, y, cv=smpls, n_jobs=n_jobs,
                                     method=method)
    else:
        if not smpls_is_partition:
            assert len(smpls) == 1
            train_index, test_index = smpls[0]
            # restrict to sorted set of test_indices
            assert np.all(np.diff(test_index) > 0), 'test_index not sorted'
            if est_params is None:
                fitted_model, _ = _fit(clone(estimator),
                                        X, y, train_index)
            elif isinstance(est_params, dict):
                fitted_model, _ = _fit(clone(estimator).set_params(**est_params),
                                        X, y, train_index)
            pred_fun = getattr(fitted_model, method)
            preds = pred_fun(X[test_index, :])
            return preds

        else:
            if method == 'predict_proba':
                y = np.asarray(y)
                le = LabelEncoder()
                y = le.fit_transform(y)

            parallel = Parallel(n_jobs=n_jobs, verbose=0, pre_dispatch='2*n_jobs')

            if est_params is None:
                fitted_models = parallel(delayed(_fit)(
                    clone(estimator), X, y, train_index, idx)
                                             for idx, (train_index, test_index) in enumerate(smpls))
            elif isinstance(est_params, dict):
                warnings.warn("Using the same (hyper-)parameters for all folds")
                fitted_models = parallel(delayed(_fit)(
                    clone(estimator).set_params(**est_params), X, y, train_index, idx)
                                             for idx, (train_index, test_index) in enumerate(smpls))
            else:
                assert len(est_params) == len(smpls), 'provide one parameter setting per fold'
                fitted_models = parallel(delayed(_fit)(
                    clone(estimator).set_params(**est_params[idx]), X, y, train_index, idx)
                                             for idx, (train_index, test_index) in enumerate(smpls))

            preds = np.zeros_like(y)
            train_preds = list()
            for idx, (train_index, test_index) in enumerate(smpls):
                assert idx == fitted_models[idx][1]
                pred_fun = getattr(fitted_models[idx][0], method)
                preds[test_index] = pred_fun(X[test_index, :])
                if return_train_preds:
                    train_preds.append(pred_fun(X[train_index, :]))

            if return_train_preds:
                return preds, train_preds
            else:
                return preds
