import numpy as np
import pytest
import math

import doubleml as dml
from doubleml.datasets import make_pliv_multiway_cluster_CKMS2021

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ._utils import draw_smpls
from ._utils_cvar_manual import fit_cvar

from doubleml.datasets import make_irm_data


@pytest.fixture(scope='module',
                params=[0, 1])
def treatment(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.25, 0.5, 0.75])
def quantile(request):
    return request.param


@pytest.fixture(scope='module',
                params=[RandomForestClassifier(max_depth=2, n_estimators=10, random_state=42),
                        LogisticRegression()])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.01, 0.05])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope="module")
def dml_cvar_fixture(generate_data_quantiles, treatment, quantile, learner,
                     dml_procedure, trimming_threshold):
    n_folds = 3

    # collect data
    (x, y, d) = generate_data_quantiles
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)
    np.random.seed(42)
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)

    np.random.seed(42)
    dml_cvar_obj = dml.DoubleMLCVAR(obj_dml_data,
                                    clone(learner), clone(learner),
                                    treatment=treatment,
                                    quantile=quantile,
                                    n_folds=n_folds,
                                    n_rep=1,
                                    trimming_threshold=trimming_threshold,
                                    draw_sample_splitting=False)

    # synchronize the sample splitting
    dml_cvar_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_cvar_obj.fit()

    np.random.seed(42)
    res_manual = fit_cvar(y, x, d, quantile,
                          clone(learner), clone(learner),
                          all_smpls, treatment, dml_procedure,
                          n_rep=1, trimming_threshold=trimming_threshold)

    res_dict = {'coef': dml_cvar_obj.coef,
                'coef_manual': res_manual['pq'],
                'se': dml_cvar_obj.se,
                'se_manual': res_manual['se']}

    return res_dict


@pytest.mark.ci
def test_dml_cvar_coef(dml_cvar_fixture):
    assert math.isclose(dml_cvar_fixture['coef'],
                        dml_cvar_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_cvar_se(dml_cvar_fixture):
    assert math.isclose(dml_cvar_fixture['se'],
                        dml_cvar_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_doubleml_cvar_exceptions():
    np.random.seed(3141)
    (x, y, d) = make_irm_data(100, 5, 2, return_type='array')
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)
    ml_g = RandomForestClassifier()
    ml_m = RandomForestClassifier()

    msg = 'Nuisance tuning not implemented for potential quantiles.'
    with pytest.raises(NotImplementedError, match=msg):
        dml_pq = dml.DoubleMLCVAR(obj_dml_data, ml_g, ml_m, treatment=1)
        _ = dml_pq.tune({'ml_g': {'n_estimators': [5, 10]},
                         'ml_m': {'n_estimators': [5, 10]},
                         'ml_m_prelim': {'n_estimators': [5, 10]}})

    msg = "Quantile has to be a float. Object of type <class 'str'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml.DoubleMLCVAR(obj_dml_data, ml_g, ml_m, treatment=1, quantile="0.4")
    msg = "Quantile has be between 0 or 1. Quantile 1.0 passed."
    with pytest.raises(ValueError, match=msg):
        _ = dml.DoubleMLCVAR(obj_dml_data, ml_g, ml_m, treatment=1, quantile=1.)

    msg = "Treatment indicator has to be an integer. Object of type <class 'str'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml.DoubleMLCVAR(obj_dml_data, ml_g, ml_m, treatment="1")
    msg = "Treatment indicator has be either 0 or 1. Treatment indicator 2 passed."
    with pytest.raises(ValueError, match=msg):
        _ = dml.DoubleMLCVAR(obj_dml_data, ml_g, ml_m, treatment=2)


@pytest.mark.ci
def test_doubleml_cluster_not_implemented_exception():
    np.random.seed(3141)
    dml_data = make_pliv_multiway_cluster_CKMS2021()
    dml_data.z_cols = None
    ml_g = RandomForestClassifier()
    ml_m = RandomForestClassifier()
    msg = 'Estimation with clustering not implemented.'
    with pytest.raises(NotImplementedError, match=msg):
        _ = dml.DoubleMLCVAR(dml_data, ml_g, ml_m, treatment=1)
