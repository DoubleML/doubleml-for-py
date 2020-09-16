import numpy as np
import pytest
import math

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

import doubleml as dml

from doubleml.tests.helper_general import get_n_datasets
from doubleml.tests.helper_plr_manual import plr_dml1, plr_dml2, fit_nuisance_plr, boot_plr


# number of datasets per dgp
n_datasets = get_n_datasets()

@pytest.fixture(scope='module',
                params = range(n_datasets))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params = [RandomForestRegressor(max_depth=2, n_estimators=10),
                          LinearRegression(),
                          Lasso(alpha=0.1)])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['IV-type', 'partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope="module")
def dml2_plr_fixture(generate_data1, idx, learner, score):
    dml_procedure = 'dml2'
    n_folds = 2

    # collect data
    data = generate_data1[idx]
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_g = clone(learner)
    ml_m = clone(learner)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], X_cols)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  dml_procedure=dml_procedure)
    dml_plr_obj.fit(se_reestimate=False)

    np.random.seed(3141)
    dml_plr_obj_reestimate_se = dml.DoubleMLPLR(obj_dml_data,
                                                ml_g, ml_m,
                                                n_folds,
                                                score=score,
                                                dml_procedure=dml_procedure)
    dml_plr_obj_reestimate_se.fit(se_reestimate=True)

    res_dict = {'se': dml_plr_obj.se,
                'se_reestimate_se': dml_plr_obj_reestimate_se.se}

    return res_dict


def test_dml2_plr_se(dml2_plr_fixture):
    assert math.isclose(dml2_plr_fixture['se'],
                        dml2_plr_fixture['se_reestimate_se'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.fixture(scope="module")
def dml1_plr_fixture(generate_data1, idx, learner, score):
    dml_procedure = 'dml1'
    n_folds = 2

    # collect data
    data = generate_data1[idx]
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    ml_g = clone(learner)
    ml_m = clone(learner)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], X_cols)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  dml_procedure=dml_procedure)
    dml_plr_obj.fit(se_reestimate=True)

    np.random.seed(3141)
    y = data['y'].values
    X = data.loc[:, X_cols].values
    d = data['d'].values
    resampling = KFold(n_splits=n_folds,
                       shuffle=True)
    smpls = [(train, test) for train, test in resampling.split(X)]

    g_hat, m_hat = fit_nuisance_plr(y, X, d,
                                    clone(learner), clone(learner), smpls)

    res_manual, se_manual = plr_dml1(y, X, d,
                                     g_hat, m_hat,
                                     smpls, score,
                                     se_reestimate=True)

    res_dict = {'se': dml_plr_obj.se,
                'se_manual': se_manual}

    return res_dict


def test_dml1_plr_se(dml1_plr_fixture):
    assert math.isclose(dml1_plr_fixture['se'],
                        dml1_plr_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)