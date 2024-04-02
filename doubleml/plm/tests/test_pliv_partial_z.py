import numpy as np
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import Lasso

import doubleml as dml

from ...tests._utils import draw_smpls
from ._utils_pliv_partial_z_manual import fit_pliv_partial_z, boot_pliv_partial_z


@pytest.fixture(scope='module',
                params=[Lasso(alpha=0.1)])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module')
def dml_pliv_partial_z_fixture(generate_data_pliv_partialZ, learner, score):
    boot_methods = ['Bayes', 'normal', 'wild']
    n_folds = 2
    n_rep_boot = 503

    # collect data
    data = generate_data_pliv_partialZ
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()
    z_cols = data.columns[data.columns.str.startswith('Z')].tolist()

    # Set machine learning methods for r
    ml_r = clone(learner)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], x_cols, z_cols)
    dml_pliv_obj = dml.DoubleMLPLIV._partialZ(obj_dml_data,
                                              ml_r,
                                              n_folds=n_folds)

    dml_pliv_obj.fit()

    np.random.seed(3141)
    y = data['y'].values
    x = data.loc[:, x_cols].values
    d = data['d'].values
    z = data.loc[:, z_cols].values
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds)

    res_manual = fit_pliv_partial_z(y, x, d, z,
                                    clone(learner),
                                    all_smpls, score)

    res_dict = {'coef': dml_pliv_obj.coef,
                'coef_manual': res_manual['theta'],
                'se': dml_pliv_obj.se,
                'se_manual': res_manual['se'],
                'boot_methods': boot_methods}

    for bootstrap in boot_methods:
        np.random.seed(3141)
        boot_t_stat = boot_pliv_partial_z(y, d, z, res_manual['thetas'], res_manual['ses'],
                                          res_manual['all_r_hat'],
                                          all_smpls, score, bootstrap, n_rep_boot)

        np.random.seed(3141)
        dml_pliv_obj.bootstrap(method=bootstrap, n_rep_boot=n_rep_boot)
        res_dict['boot_t_stat' + bootstrap] = dml_pliv_obj.boot_t_stat
        res_dict['boot_t_stat' + bootstrap + '_manual'] = boot_t_stat.reshape(-1, 1, 1)

    return res_dict


def test_dml_pliv_coef(dml_pliv_partial_z_fixture):
    assert math.isclose(dml_pliv_partial_z_fixture['coef'],
                        dml_pliv_partial_z_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_se(dml_pliv_partial_z_fixture):
    assert math.isclose(dml_pliv_partial_z_fixture['se'],
                        dml_pliv_partial_z_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_boot(dml_pliv_partial_z_fixture):
    for bootstrap in dml_pliv_partial_z_fixture['boot_methods']:
        assert np.allclose(dml_pliv_partial_z_fixture['boot_t_stat' + bootstrap],
                           dml_pliv_partial_z_fixture['boot_t_stat' + bootstrap + '_manual'],
                           rtol=1e-9, atol=1e-4)
