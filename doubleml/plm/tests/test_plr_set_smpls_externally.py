import numpy as np
import pytest
import math

from sklearn.linear_model import LinearRegression

import doubleml as dml

from ...tests._utils import _clone


@pytest.fixture(scope='module',
                params=[LinearRegression()])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['IV-type', 'partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_smpls_fixture(generate_data1, learner, score, n_rep):
    n_folds = 3

    # collect data
    data = generate_data1
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for l, m & g
    ml_l = _clone(learner)
    ml_m = _clone(learner)
    if score == 'IV-type':
        ml_g = _clone(learner)
    else:
        ml_g = None

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_l, ml_m, ml_g,
                                  n_folds,
                                  n_rep,
                                  score)

    dml_plr_obj.fit()

    smpls = dml_plr_obj.smpls

    dml_plr_obj2 = dml.DoubleMLPLR(obj_dml_data,
                                   ml_l, ml_m, ml_g,
                                   score=score,
                                   draw_sample_splitting=False)
    dml_plr_obj2.set_sample_splitting(smpls)
    dml_plr_obj2.fit()

    res_dict = {'coef': dml_plr_obj.coef,
                'coef2': dml_plr_obj2.coef,
                'se': dml_plr_obj.se,
                'se2': dml_plr_obj2.se}

    return res_dict


@pytest.mark.ci
def test_dml_plr_coef(dml_plr_smpls_fixture):
    assert math.isclose(dml_plr_smpls_fixture['coef'],
                        dml_plr_smpls_fixture['coef2'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_se(dml_plr_smpls_fixture):
    assert math.isclose(dml_plr_smpls_fixture['se'],
                        dml_plr_smpls_fixture['se2'],
                        rel_tol=1e-9, abs_tol=1e-4)
