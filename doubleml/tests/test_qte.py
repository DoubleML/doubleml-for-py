import numpy as np
import pytest
import math

import doubleml as dml
from doubleml.datasets import make_pliv_multiway_cluster_CKMS2021

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ._utils import draw_smpls
from ._utils_qte_manual import fit_qte


@pytest.fixture(scope='module',
                params=[RandomForestClassifier(max_depth=2, n_estimators=10, random_state=42),
                        LogisticRegression()])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope="module")
def dml_qte_fixture(generate_data_quantiles, learner, dml_procedure):
    n_folds = 3

    # collect data
    (x, y, d) = generate_data_quantiles
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)

    # Set machine learning methods for g & m
    ml_g = clone(learner)
    ml_m = clone(learner)

    np.random.seed(42)
    quantiles = [0.25, 0.5, 0.75]
    dml_qte_obj = dml.DoubleMLQTE(obj_dml_data,
                                  ml_g, ml_m,
                                  quantiles=quantiles,
                                  n_folds=n_folds,
                                  dml_procedure=dml_procedure,
                                  trimming_threshold=1e-12)

    dml_qte_obj.fit()

    np.random.seed(42)
    n_obs = len(y)
    all_smpls = draw_smpls(n_obs, n_folds, n_rep=1)
    res_manual = fit_qte(y, x, d, quantiles, ml_g, ml_g, all_smpls,
                         n_rep=1, dml_procedure=dml_procedure,
                         trimming_rule='truncate', trimming_threshold=1e-12, h=None,
                         normalize=True, draw_sample_splitting=True)

    res_dict = {'coef': dml_qte_obj.coef,
                'coef_manual': res_manual['qte'],
                'se': dml_qte_obj.se,
                'se_manual': res_manual['se']}

    return res_dict


@pytest.mark.ci
def test_dml_qte_coef(dml_qte_fixture):
    assert all(np.isclose(dml_qte_fixture['coef'],
                          dml_qte_fixture['coef_manual'],
                          atol=1e-9, rtol=1e-4))



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
