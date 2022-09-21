import numpy as np
import pytest

from sklearn.linear_model import Lasso, ElasticNet

import doubleml as dml

from ._utils import _clone


@pytest.fixture(scope='module',
                params=[Lasso(),
                        ElasticNet()])
def learner_g(request):
    return request.param


@pytest.fixture(scope='module',
                params=[Lasso(),
                        ElasticNet()])
def learner_m(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def tune_on_folds(request):
    return request.param


def get_par_grid(learner):
    if learner.__class__ == Lasso:
        par_grid = {'alpha': np.linspace(0.05, .95, 7)}
    else:
        assert learner.__class__ == ElasticNet
        par_grid = {'l1_ratio': [.1, .5, .7, .9, .95, .99, 1], 'alpha': np.linspace(0.05, 1., 7)}
    return par_grid


@pytest.fixture(scope="module")
def dml_pcop_tune_fixture(generate_data2, learner_g, learner_m, tune_on_folds):
    n_folds_tune = 4

    n_folds = 2

    # collect data
    obj_dml_data = generate_data2

    # Set machine learning methods for l & m
    ml_g = _clone(learner_g)
    ml_m = _clone(learner_m)

    np.random.seed(3141)
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds=n_folds,
                                  score='partialling out')

    # tune hyperparameters
    par_grid = {'ml_l': get_par_grid(learner_g),
                'ml_m': get_par_grid(learner_m)}
    tune_res_plr = dml_plr_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune,
                                    return_tune_res=True)

    dml_data_pcop = dml.DoubleMLPartialDependenceData(obj_dml_data.data, y_col='y', z_col='d')

    np.random.seed(3141)
    dml_pcop = dml.DoubleMLPartialCopula(dml_data_pcop,
                                         'Gaussian',
                                         ml_g, ml_m,
                                         n_folds=n_folds)

    # tune hyperparameters
    par_grid = {'ml_g': get_par_grid(learner_g),
                'ml_m': get_par_grid(learner_m)}
    tune_res_pcop = dml_pcop.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune,
                                  return_tune_res=True)

    np.random.seed(3141)
    dml_pcorr = dml.DoubleMLPartialCorr(dml_data_pcop,
                                        ml_g, ml_m,
                                        n_folds=n_folds)

    # tune hyperparameters
    par_grid = {'ml_g': get_par_grid(learner_g),
                'ml_m': get_par_grid(learner_m)}
    tune_res_pcorr = dml_pcorr.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune,
                                    return_tune_res=True)

    if tune_on_folds:
        res_dict = {'params_plr_ml_l': tune_res_plr[0][0]['params']['ml_l'],
                    'params_plr_ml_m': tune_res_plr[0][0]['params']['ml_m'],
                    'params_pcop_ml_g': tune_res_pcop[0][0]['params']['ml_g'],
                    'params_pcop_ml_m': tune_res_pcop[0][0]['params']['ml_m'],
                    'params_pcorr_ml_g': tune_res_pcorr[0][0]['params']['ml_g'],
                    'params_pcorr_ml_m': tune_res_pcorr[0][0]['params']['ml_m']}
    else:
        res_dict = {'params_plr_ml_l': tune_res_plr[0]['params']['ml_l'],
                    'params_plr_ml_m': tune_res_plr[0]['params']['ml_m'],
                    'params_pcop_ml_g': tune_res_pcop[0]['params']['ml_g'],
                    'params_pcop_ml_m': tune_res_pcop[0]['params']['ml_m'],
                    'params_pcorr_ml_g': tune_res_pcorr[0]['params']['ml_g'],
                    'params_pcorr_ml_m': tune_res_pcorr[0]['params']['ml_m']}

    return res_dict


@pytest.mark.ci
def test_dml_pcop_tune_ml_g(dml_pcop_tune_fixture):
    for i_par in range(len(dml_pcop_tune_fixture['params_pcop_ml_g'])):
        assert dml_pcop_tune_fixture['params_plr_ml_l'][i_par] == dml_pcop_tune_fixture['params_pcop_ml_g'][i_par]
        assert dml_pcop_tune_fixture['params_pcorr_ml_g'][i_par] == dml_pcop_tune_fixture['params_pcop_ml_g'][i_par]


@pytest.mark.ci
def test_dml_pcop_tune_ml_m(dml_pcop_tune_fixture):
    for i_par in range(len(dml_pcop_tune_fixture['params_pcop_ml_m'])):
        assert dml_pcop_tune_fixture['params_plr_ml_m'][i_par] == dml_pcop_tune_fixture['params_pcop_ml_m'][i_par]
        assert dml_pcop_tune_fixture['params_pcorr_ml_m'][i_par] == dml_pcop_tune_fixture['params_pcop_ml_m'][i_par]
