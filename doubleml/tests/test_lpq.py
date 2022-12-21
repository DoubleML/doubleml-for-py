import numpy as np
import pytest
import math

import doubleml as dml
from doubleml.datasets import make_pliv_multiway_cluster_CKMS2021

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ._utils import draw_smpls
from ._utils_lpq_manual import fit_lpq

from doubleml.datasets import make_iivm_data


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
def dml_lpq_fixture(generate_data_local_quantiles, treatment, quantile, learner,
                    dml_procedure, trimming_threshold):
    n_folds = 3

    # collect data
    (x, y, d, z) = generate_data_local_quantiles
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d, z)
    np.random.seed(42)
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1, groups=d)

    np.random.seed(42)
    dml_lpq_obj = dml.DoubleMLLPQ(obj_dml_data,
                                  clone(learner),
                                  treatment=treatment,
                                  quantile=quantile,
                                  n_folds=n_folds,
                                  n_rep=1,
                                  dml_procedure=dml_procedure,
                                  trimming_threshold=trimming_threshold,
                                  draw_sample_splitting=False)

    # synchronize the sample splitting
    dml_lpq_obj.set_sample_splitting(all_smpls=all_smpls)
    dml_lpq_obj.fit()

    np.random.seed(42)
    res_manual = fit_lpq(y, x, d, z, quantile, clone(learner),
                         all_smpls, treatment, dml_procedure,
                         n_rep=1, trimming_threshold=trimming_threshold)

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


@pytest.mark.ci
def test_doubleml_lpq_exceptions():
    np.random.seed(3141)
    data = make_iivm_data(100, 5, 2, return_type='DataFrame')
    obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_cols='z')
    ml_m = RandomForestClassifier()

    msg = 'Nuisance tuning not implemented for potential quantiles.'
    with pytest.raises(NotImplementedError, match=msg):
        dml_lpq = dml.DoubleMLLPQ(obj_dml_data, ml_m, treatment=1)
        _ = dml_lpq.tune({'ml_pi_z': {'n_estimators': [5, 10]},
                          'ml_pi_du_z0': {'n_estimators': [5, 10]},
                          'ml_pi_du_z1': {'n_estimators': [5, 10]},
                          'ml_pi_d_z0': {'n_estimators': [5, 10]},
                          'ml_pi_d_z1': {'n_estimators': [5, 10]}})

    msg = "Quantile has to be a float. Object of type <class 'str'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml.DoubleMLLPQ(obj_dml_data, ml_m, treatment=1, quantile="0.4")
    msg = "Quantile has be between 0 or 1. Quantile 1.0 passed."
    with pytest.raises(ValueError, match=msg):
        _ = dml.DoubleMLLPQ(obj_dml_data, ml_m, treatment=1, quantile=1.)

    msg = "Treatment indicator has to be an integer. Object of type <class 'str'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml.DoubleMLLPQ(obj_dml_data, ml_m, treatment="1")
    msg = "Treatment indicator has be either 0 or 1. Treatment indicator 2 passed."
    with pytest.raises(ValueError, match=msg):
        _ = dml.DoubleMLLPQ(obj_dml_data, ml_m, treatment=2)

    msg = "Bandwidth has to be a float. Object of type <class 'str'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml.DoubleMLLPQ(obj_dml_data, ml_m, treatment=1, h="0.1")
    msg = "Bandwidth has be positive. Bandwidth -0.1 passed."
    with pytest.raises(ValueError, match=msg):
        _ = dml.DoubleMLLPQ(obj_dml_data, ml_m, treatment=1, h=-0.1)

    msg = "Normalization indicator has to be boolean. Object of type <class 'int'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml.DoubleMLLPQ(obj_dml_data, ml_m, treatment=1, normalize=1)


@pytest.mark.ci
def test_doubleml_cluster_not_implemented_exception():
    np.random.seed(3141)
    dml_data = make_pliv_multiway_cluster_CKMS2021()
    ml_g = RandomForestClassifier()
    msg = 'Estimation with clustering not implemented.'
    with pytest.raises(NotImplementedError, match=msg):
        _ = dml.DoubleMLLPQ(dml_data, ml_g, treatment=1)
