import numpy as np
import pytest

from sklearn.model_selection import KFold, train_test_split

from sklearn.linear_model import Lasso

from doubleml.tests.helper_general import get_n_datasets
from doubleml.tests.helper_dml_cv_predict import _dml_cv_predict_ut_version
from doubleml._helper import _dml_cv_predict

# number of datasets per dgp
n_datasets = get_n_datasets()


@pytest.fixture(scope='module',
                params=range(n_datasets))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def cross_fit(request):
    return request.param


@pytest.fixture(scope='module',
                params=[None, 'global', 'per_fold'])
def params(request):
    return request.param


@pytest.fixture(scope='module')
def cv_predict_fixture(generate_data_cv_predict, idx, cross_fit, params):
    n_folds = 4
    # collect data
    (x, y) = generate_data_cv_predict[idx]

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
        est_params = {'alpha': 0.5}
    elif params == 'per_fold':
        if cross_fit:
            est_params = [{'alpha': np.random.uniform()} for i in range(n_folds)]
        else:
            est_params = {'alpha': 1.}

    preds = _dml_cv_predict(Lasso(), x, y, smpls, est_params=est_params)
    preds_ut = _dml_cv_predict_ut_version(Lasso(), x, y, smpls, est_params=est_params)

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
