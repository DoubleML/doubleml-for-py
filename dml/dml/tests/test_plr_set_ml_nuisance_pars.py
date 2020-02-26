import numpy as np
import pytest
import math
import scipy

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

from dml.double_ml_data import DoubleMLData
from dml.double_ml_plr import DoubleMLPLR

from dml.tests.helper_general import get_n_datasets
from dml.tests.helper_plr_manual import plr_dml1, plr_dml2, fit_nuisance_plr, boot_plr


# number of datasets per dgp
n_datasets = get_n_datasets()

@pytest.fixture(scope='module',
                params = range(n_datasets))
def idx(request):
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
def dml_plr_fixture(generate_data1, idx, inf_model, dml_procedure):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 502

    alpha = 0.05

    learner = Lasso(alpha=alpha)
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

    np.random.seed(3141)
    learner = Lasso()
    # Set machine learning methods for m & g
    ml_learners = {'ml_m': clone(learner),
                   'ml_g': clone(learner)}

    dml_plr_obj_ext_set_par = DoubleMLPLR(n_folds,
                                          ml_learners,
                                          dml_procedure,
                                          inf_model)
    dml_plr_obj_ext_set_par.set_ml_nuisance_params({'g_params': {'alpha': alpha},
                                                    'm_params': {'alpha': alpha}})
    dml_plr_obj_ext_set_par.fit(obj_dml_data)

    
    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': dml_plr_obj_ext_set_par.coef,
                'se': dml_plr_obj.se,
                'se_manual': dml_plr_obj_ext_set_par.se,
                'boot_methods': boot_methods}
    
    for bootstrap in boot_methods:
        np.random.seed(314122)
        dml_plr_obj.bootstrap(method = bootstrap, n_rep=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_plr_obj.boot_coef
        
        np.random.seed(314122)
        dml_plr_obj_ext_set_par.bootstrap(method = bootstrap, n_rep=n_rep_boot)
        res_dict['boot_coef' + bootstrap + '_manual'] = dml_plr_obj_ext_set_par.boot_coef
    
    return res_dict


@pytest.mark.filterwarnings("ignore:Using the same")
def test_dml_plr_coef(dml_plr_fixture):
    assert math.isclose(dml_plr_fixture['coef'],
                        dml_plr_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.filterwarnings("ignore:Using the same")
def test_dml_plr_se(dml_plr_fixture):
    assert math.isclose(dml_plr_fixture['se'],
                        dml_plr_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.filterwarnings("ignore:Using the same")
def test_dml_plr_boot(dml_plr_fixture):
    for bootstrap in dml_plr_fixture['boot_methods']:
        assert np.allclose(dml_plr_fixture['boot_coef' + bootstrap],
                           dml_plr_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)

