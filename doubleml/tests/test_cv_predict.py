import numpy as np
import pytest

from sklearn.model_selection import KFold, train_test_split

from sklearn.linear_model import Lasso, LogisticRegression

from ._utils_dml_cv_predict import _dml_cv_predict_ut_version
from doubleml._utils import _dml_cv_predict


@pytest.fixture(scope='module',
                params=[True, False])
def cross_fit(request):
    return request.param


@pytest.fixture(scope='module',
                params=[None, 'global', 'per_fold'])
def params(request):
    return request.param


@pytest.fixture(scope='module')
def cv_predict_fixture(generate_data_cv_predict, cross_fit, params):
    n_folds = 4
    # collect data
    (x, y, classifier) = generate_data_cv_predict

    if classifier:
        method = 'predict_proba'
    else:
        method = 'predict'

    if cross_fit:
        smpls = [(train, test) for train, test in KFold(n_splits=n_folds,
                                                        shuffle=True).split(x)]
    else:
        n_obs = len(y)
        smpls = train_test_split(np.arange(n_obs), test_size=0.23)
        smpls = [[np.sort(x) for x in smpls]]  # only sorted indices are supported

    if params is None:
        est_params = None
    elif params == 'global':
        if method == 'predict_proba':
            est_params = {'C': 0.5}
        else:
            est_params = {'alpha': 0.5}
    else:
        assert params == 'per_fold'
        if method == 'predict_proba':
            if cross_fit:
                est_params = [{'C': np.random.uniform()} for i in range(n_folds)]
            else:
                est_params = {'C': 1.}
        else:
            if cross_fit:
                est_params = [{'alpha': np.random.uniform()} for i in range(n_folds)]
            else:
                est_params = {'alpha': 1.}

    if method == 'predict_proba':
        preds = _dml_cv_predict(LogisticRegression(), x, y, smpls,
                                est_params=est_params, method=method)
        preds_ut = _dml_cv_predict_ut_version(LogisticRegression(), x, y, smpls,
                                              est_params=est_params, method=method)[:, 1]
    else:
        preds = _dml_cv_predict(Lasso(), x, y, smpls, est_params=est_params, method=method)
        preds_ut = _dml_cv_predict_ut_version(Lasso(), x, y, smpls, est_params=est_params, method=method)

    res_dict = {'preds': preds,
                'preds_ut': preds_ut}

    return res_dict


@pytest.mark.ci
def test_cv_predict(cv_predict_fixture):
    ind_nan_preds = np.isnan(cv_predict_fixture['preds'])
    ind_nan_preds_ut = np.isnan(cv_predict_fixture['preds_ut'])
    assert np.array_equal(ind_nan_preds, ind_nan_preds_ut)
    assert np.allclose(cv_predict_fixture['preds'][~ind_nan_preds],
                       cv_predict_fixture['preds_ut'][~ind_nan_preds],
                       rtol=1e-9, atol=1e-4)
