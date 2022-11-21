import numpy as np
import pytest
import math
import scipy.stats as sps

import doubleml as dml
from doubleml.datasets import make_pliv_multiway_cluster_CKMS2021

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from ._utils import draw_smpls
from ._utils_pq_manual import fit_pq


@pytest.fixture(scope='module',
                params=[0, 1])
def treatment(request):
    return request.param

@pytest.fixture(scope='module',
                params=[0.25, 0.5, 0.75])
def quantile(request):
    return request.param

@pytest.fixture(scope='module',
                params=[RandomForestClassifier(max_depth=2, n_estimators=10),
                        AdaBoostClassifier(n_estimators=10),
                        DecisionTreeClassifier(max_depth=5),
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
def dml_pq_fixture(generate_data_quantiles, treatment, quantile, learner,
                   dml_procedure, trimming_threshold):
    n_folds = 3

    # collect data
    (x, y, d) = generate_data_quantiles
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)

    # True Quantile (Remark that this is dependent on location and scale)
    error_quantile = sps.norm(loc=0, scale=1).ppf(quantile)
    if treatment == 0:
        pq = 0 + error_quantile
    else:
        pq = 2 + np.sqrt(0.5) * error_quantile

    # Set machine learning methods for g & m
    ml_g = clone(learner)
    ml_m = clone(learner)

    np.random.seed(42)
    dml_pq_obj = dml.DoubleMLPQ(obj_dml_data,
                                ml_g, ml_m,
                                treatment=treatment,
                                tau=quantile,
                                n_folds=n_folds,
                                dml_procedure=dml_procedure,
                                trimming_threshold=trimming_threshold)

    np.random.seed(42)
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds)
    res_manual = fit_pq(y, x, d, quantile,
                        clone(learner), clone(learner),
                        all_smpls,treatment, dml_procedure,
                        n_rep=1, trimming_threshold=trimming_threshold)

    res_dict = {'coef': dml_pq_obj.coef,
                'coef_manual': res_manual['pq'],
                'se': dml_pq_obj.se,
                'se_manual': res_manual['se']}

    return res_dict


@pytest.mark.ci
def test_dml_irm_coef(dml_pq_fixture):
    assert math.isclose(dml_pq_fixture['coef'],
                        dml_pq_fixture['coef_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)

@pytest.mark.ci
def test_dml_irm_se(dml_irm_fixture):
    assert math.isclose(dml_pq_fixture['se'],
                        dml_pq_fixture['se_manual'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.mark.ci
def test_dml_plr_coef(dml_pq_fixture):
    assert math.isclose(1,
                        1,
                        rel_tol=1e-9, abs_tol=1e-4)



@pytest.mark.ci
def test_doubleml_cluster_not_implemented_exception():
    np.random.seed(3141)
    dml_data = make_pliv_multiway_cluster_CKMS2021()
    dml_data.z_cols = None
    ml_g = RandomForestClassifier()
    ml_m = RandomForestClassifier()
    msg = 'Estimation with clustering not implemented.'
    with pytest.raises(NotImplementedError, match=msg):
        _ = dml.DoubleMLPQ(dml_data, ml_g, ml_m, treatment=1)
