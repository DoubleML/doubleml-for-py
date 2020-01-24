import numpy as np
import pytest
import math

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

from dml.double_ml_data import DoubleMLData
from dml.double_ml_plr import DoubleMLPLR

from dml.tests.helper_general import get_n_datasets


# number of datasets per dgp
n_datasets = get_n_datasets()

@pytest.fixture(scope='module',
                params = range(n_datasets))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params = [LinearRegression()])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['IV-type', 'DML2018'])
def inf_model(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_smpls_fixture(generate_data1, idx, learner, inf_model, dml_procedure):
    n_folds = 3
    n_rep_boot = 371
    
    # Set machine learning methods for m & g
    ml_learners = {'ml_m': clone(learner),
                   'ml_g': clone(learner)}
    
    dml_plr_obj = DoubleMLPLR(n_folds,
                              ml_learners,
                              dml_procedure,
                              inf_model)
    data = generate_data1[idx]
    np.random.seed(3141)
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()
    obj_dml_data = DoubleMLData(data, X_cols, 'y', ['d'])
    dml_plr_obj.fit(obj_dml_data)

    smpls = dml_plr_obj.smpls

    n_folds = 3
    dml_plr_obj2 = DoubleMLPLR(n_folds,
                               ml_learners,
                               dml_procedure,
                               inf_model)
    dml_plr_obj2.set_samples(smpls)
    dml_plr_obj2.fit(obj_dml_data)
    
    res_dict = {'coef': dml_plr_obj.coef,
                'coef2': dml_plr_obj2.coef,
                'se': dml_plr_obj.se,
                'se2': dml_plr_obj2.se}

    return res_dict


def test_dml_plr_coef(dml_plr_smpls_fixture):
    assert math.isclose(dml_plr_smpls_fixture['coef'],
                        dml_plr_smpls_fixture['coef2'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_plr_se(dml_plr_smpls_fixture):
    assert math.isclose(dml_plr_smpls_fixture['se'],
                        dml_plr_smpls_fixture['se2'],
                        rel_tol=1e-9, abs_tol=1e-4)

