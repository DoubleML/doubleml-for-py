import numpy as np
import pytest
import math

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import doubleml as dml

from ._utils import draw_smpls
from ._utils_lpq_manual import fit_lpq, tune_nuisance_lpq


@pytest.fixture(scope='module',
                params=[0])
def treatment(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.5])
def quantile(request):
    return request.param


@pytest.fixture(scope='module',
                params=[RandomForestClassifier(max_depth=2, n_estimators=5, random_state=42)])
def learner_pi(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def tune_on_folds(request):
    return request.param


def get_par_grid(learner):
    if learner.__class__ in [RandomForestClassifier]:
        par_grid = {'n_estimators': [5, 10, 20]}
    else:
        assert learner.__class__ in [LogisticRegression]
        par_grid = {'C': np.logspace(-4, 2, 10)}
    return par_grid


@pytest.fixture(scope='module')
def dml_lpq_fixture(generate_data_local_quantiles, treatment, quantile, learner_pi, dml_procedure, normalize_ipw,
                    tune_on_folds):
    par_grid = {'ml_pi_z': get_par_grid(learner_pi),
                'ml_pi_d_z0': get_par_grid(learner_pi),
                'ml_pi_d_z1': get_par_grid(learner_pi),
                'ml_pi_du_z0': get_par_grid(learner_pi),
                'ml_pi_du_z1': get_par_grid(learner_pi)}
    n_folds_tune = 4
    n_folds = 2

    # collect data
    (x, y, d, z) = generate_data_local_quantiles
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d, z)
    np.random.seed(42)
    n_obs = len(y)
    strata = d + 2 * z
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=strata)
    smpls = all_smpls[0]

    np.random.seed(42)
    dml_lpq_obj = dml.DoubleMLLPQ(obj_dml_data,
                                  clone(learner_pi),
                                  treatment=treatment,
                                  quantile=quantile,
                                  n_folds=n_folds,
                                  n_rep=1,
                                  dml_procedure=dml_procedure,
                                  normalize_ipw=normalize_ipw,
                                  trimming_threshold=0.01,
                                  draw_sample_splitting=False)

    # synchronize the sample splitting
    dml_lpq_obj.set_sample_splitting(all_smpls=all_smpls)
    # tune hyperparameters
    np.random.seed(42)
    tune_res = dml_lpq_obj.tune(par_grid, tune_on_folds=tune_on_folds, n_folds_tune=n_folds_tune, return_tune_res=False)
    assert isinstance(tune_res, dml.DoubleMLLPQ)

    np.random.seed(42)
    dml_lpq_obj.fit()

    np.random.seed(42)
    if tune_on_folds:
        pi_z_params, pi_d_z0_params, pi_d_z1_params, \
            pi_du_z0_params, pi_du_z1_params = tune_nuisance_lpq(y, x, d, z,
                                                                 clone(learner_pi),
                                                                 clone(learner_pi), clone(learner_pi),
                                                                 clone(learner_pi), clone(learner_pi),
                                                                 smpls, treatment, quantile, n_folds_tune,
                                                                 par_grid['ml_pi_z'],
                                                                 par_grid['ml_pi_d_z0'], par_grid['ml_pi_d_z1'],
                                                                 par_grid['ml_pi_du_z0'], par_grid['ml_pi_du_z1'])
    else:
        xx = [(np.arange(len(y)), np.array([]))]
        pi_z_params, pi_d_z0_params, pi_d_z1_params, \
            pi_du_z0_params, pi_du_z1_params = tune_nuisance_lpq(y, x, d, z,
                                                                 clone(learner_pi),
                                                                 clone(learner_pi), clone(learner_pi),
                                                                 clone(learner_pi), clone(learner_pi),
                                                                 xx, treatment, quantile, n_folds_tune,
                                                                 par_grid['ml_pi_z'],
                                                                 par_grid['ml_pi_d_z0'], par_grid['ml_pi_d_z1'],
                                                                 par_grid['ml_pi_du_z0'], par_grid['ml_pi_du_z1'])

        pi_z_params = pi_z_params * n_folds
        pi_d_z0_params = pi_d_z0_params * n_folds
        pi_d_z1_params = pi_d_z1_params * n_folds
        pi_du_z0_params = pi_du_z0_params * n_folds
        pi_du_z1_params = pi_du_z1_params * n_folds

    np.random.seed(42)
    res_manual = fit_lpq(y, x, d, z,
                         quantile=quantile,
                         learner_m=clone(learner_pi),
                         all_smpls=all_smpls,
                         treatment=treatment,
                         dml_procedure=dml_procedure,
                         n_rep=1, trimming_threshold=0.01,
                         normalize_ipw=normalize_ipw,
                         pi_z_params=pi_z_params,
                         pi_d_z0_params=pi_d_z0_params, pi_d_z1_params=pi_d_z1_params,
                         pi_du_z0_params=pi_du_z0_params, pi_du_z1_params=pi_du_z1_params)

    res_dict = {'coef': dml_lpq_obj.coef,
                'coef_manual': res_manual['lpq'],
                'se': dml_lpq_obj.se,
                'se_manual': res_manual['se']}

    return res_dict


@pytest.mark.ci
def test_dml_lpq_coef(dml_lpq_fixture):
    assert math.isclose(dml_lpq_fixture['coef'],
                        dml_lpq_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_lpq_se(dml_lpq_fixture):
    assert math.isclose(dml_lpq_fixture['se'],
                        dml_lpq_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)
