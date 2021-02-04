import numpy as np
import pytest
import math

from sklearn.base import clone
from sklearn.linear_model import Lasso

import doubleml as dml


@pytest.fixture(scope='module',
                params=['IV-type', 'partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope="module")
def dml_plr_fixture(generate_data1, score, dml_procedure):
    boot_methods = ['normal']
    n_folds = 2
    n_rep_boot = 502

    # collect data
    data = generate_data1

    alpha = 0.05
    learner = Lasso(alpha=alpha)
    # Set machine learning methods for m & g
    ml_g = clone(learner)
    ml_m = clone(learner)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'])
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  dml_procedure=dml_procedure)

    dml_plr_obj.fit()

    np.random.seed(3141)
    learner = Lasso()
    # Set machine learning methods for m & g
    ml_g = clone(learner)
    ml_m = clone(learner)

    dml_plr_obj_ext_set_par = dml.DoubleMLPLR(obj_dml_data,
                                              ml_g, ml_m,
                                              n_folds,
                                              score=score,
                                              dml_procedure=dml_procedure)
    dml_plr_obj_ext_set_par.set_ml_nuisance_params('ml_g', 'd', {'alpha': alpha})
    dml_plr_obj_ext_set_par.set_ml_nuisance_params('ml_m', 'd', {'alpha': alpha})
    dml_plr_obj_ext_set_par.fit()

    res_dict = {'coef': dml_plr_obj.coef,
                'coef_manual': dml_plr_obj_ext_set_par.coef,
                'se': dml_plr_obj.se,
                'se_manual': dml_plr_obj_ext_set_par.se,
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(314122)
        dml_plr_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap] = dml_plr_obj.boot_coef
        res_dict['boot_t_stat' + bootstrap] = dml_plr_obj.boot_t_stat

        np.random.seed(314122)
        dml_plr_obj_ext_set_par.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_coef' + bootstrap + '_manual'] = dml_plr_obj_ext_set_par.boot_coef
        res_dict['boot_t_stat' + bootstrap + '_manual'] = dml_plr_obj_ext_set_par.boot_t_stat

    return res_dict


@pytest.mark.ci
@pytest.mark.filterwarnings("ignore:Using the same")
def test_dml_plr_coef(dml_plr_fixture):
    assert math.isclose(dml_plr_fixture['coef'],
                        dml_plr_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
@pytest.mark.filterwarnings("ignore:Using the same")
def test_dml_plr_se(dml_plr_fixture):
    assert math.isclose(dml_plr_fixture['se'],
                        dml_plr_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
@pytest.mark.filterwarnings("ignore:Using the same")
def test_dml_plr_boot(dml_plr_fixture):
    for bootstrap in dml_plr_fixture['boot_methods']:
        assert np.allclose(dml_plr_fixture['boot_coef' + bootstrap],
                           dml_plr_fixture['boot_coef' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
        assert np.allclose(dml_plr_fixture['boot_t_stat' + bootstrap],
                           dml_plr_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
