import numpy as np

import warnings

import scipy.sparse as sp
from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.utils.validation import _num_samples
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection._validation import _fit_and_predict, _check_is_permutation


def _dml_cv_predict_ut_version(estimator, X, y, smpls=None,
                    n_jobs=None, est_params=None, method='predict'):
    # this is an adapted version of the sklearn function cross_val_predict which allows to set fold-specific parameters
    # original https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_validation.py

    test_indices = np.concatenate([test_index for _, test_index in smpls])
    smpls_is_partition = _check_is_permutation(test_indices, _num_samples(X))

    if not smpls_is_partition:
        assert len(smpls) == 1
        train_index, test_index = smpls[0]
        # set some defaults aligned with cross_val_predict
        fit_params = None
        verbose = 0
        predictions = np.full(len(y), np.nan)
        if est_params is None:
            xx, test_indices = _fit_and_predict(clone(estimator),
                                                X, y, train_index, test_index, verbose, fit_params, method)
        elif isinstance(est_params, dict):
            xx, test_indices = _fit_and_predict(clone(estimator).set_params(**est_params),
                                                X, y, train_index, test_index, verbose, fit_params, method)

        # implementation is (also at other parts) restricted to a sorted set of test_indices, but this could be fixed
        # inv_test_indices = np.argsort(test_indices)
        assert np.all(np.diff(test_indices)>0), 'test_indices not sorted'
        predictions[test_indices] = xx
        return predictions

    # set some defaults aligned with cross_val_predict
    fit_params = None
    verbose = 0
    pre_dispatch = '2*n_jobs'

    encode = (method == 'predict_proba')

    if encode:
        y = np.asarray(y)
        le = LabelEncoder()
        y = le.fit_transform(y)

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    # FixMe: Find a better way to handle the different combinations of paramters and smpls_is_partition
    if est_params is None:
        prediction_blocks = parallel(delayed(_fit_and_predict)(
            estimator,
            X, y, train_index, test_index, verbose, fit_params, method)
                                     for idx, (train_index, test_index) in enumerate(smpls))
    elif isinstance(est_params, dict):
        # if no fold-specific parameters we redirect to the standard method
        # warnings.warn("Using the same (hyper-)parameters for all folds")
        prediction_blocks = parallel(delayed(_fit_and_predict)(
            clone(estimator).set_params(**est_params),
            X, y, train_index, test_index, verbose, fit_params, method)
                                     for idx, (train_index, test_index) in enumerate(smpls))
    else:
        assert len(est_params) == len(smpls), 'provide one parameter setting per fold'
        prediction_blocks = parallel(delayed(_fit_and_predict)(
            clone(estimator).set_params(**est_params[idx]),
            X, y, train_index, test_index, verbose, fit_params, method)
            for idx, (train_index, test_index) in enumerate(smpls))

    # Concatenate the predictions
    predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
    test_indices = np.concatenate([indices_i
                                   for _, indices_i in prediction_blocks])

    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('_dml_cross_val_predict only works for partitions')

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    elif encode and isinstance(predictions[0], list):
        n_labels = y.shape[1]
        concat_pred = []
        for i_label in range(n_labels):
            label_preds = np.concatenate([p[i_label] for p in predictions])
            concat_pred.append(label_preds)
        predictions = concat_pred
    else:
        predictions = np.concatenate(predictions)

    if isinstance(predictions, list):
        return [p[inv_test_indices] for p in predictions]
    else:
        return predictions[inv_test_indices]
