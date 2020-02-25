import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import cross_val_predict

import scipy.sparse as sp
from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.utils.validation import _num_samples
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection._validation import _fit_and_predict, _check_is_permutation


def assure_2d_array(x):
    if x.ndim == 1:
        x = x.reshape(-1,1)
    elif x.ndim > 2:
        raise ValueError('Only one- or two-dimensional arrays are allowed')
    return x


def check_binary_vector(x, variable_name=''):
    # assure D binary
    assert type_of_target(x) == 'binary', 'variable ' + variable_name  + ' must be binary'
    
    if np.any(np.power(x,2) - x != 0):
        raise ValueError('variable ' + variable_name  + ' must be binary with values 0 and 1')

def _dml_cross_val_predict(estimator, X, y, smpls=None,
                           n_jobs=None, fit_params=None, method='predict'):
    # this is an adapted version of the sklearn function cross_val_predict which allows to set fold-specific parameters
    # original https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_validation.py

    if (fit_params is None) or isinstance(fit_params, dict):
        # if there are no parameters set or no fold-specific parameters we redirect to the standard method
        return cross_val_predict(estimator, X, y, cv=smpls, n_jobs=n_jobs, method=method, fit_params=fit_params)

    # set some defaults aligned with cross_val_predict
    verbose = 0
    pre_dispatch = '2*n_jobs'

    assert len(fit_params) == len(smpls), 'provide one parameter setting per fold'

    encode = (method == 'predict_proba')

    if encode:
        y = np.asarray(y)
        le = LabelEncoder()
        y = le.fit_transform(y)

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    prediction_blocks = parallel(delayed(_fit_and_predict)(
        clone(estimator), X, y, train_index, test_index, verbose, fit_params[idx], method)
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
