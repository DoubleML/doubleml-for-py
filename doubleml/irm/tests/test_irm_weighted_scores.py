import pytest
import numpy as np

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import doubleml as dml
from doubleml.utils._estimation import _normalize_ipw


def old_score_elements(y, d, g_hat0, g_hat1, m_hat, score, normalize_ipw):
    # fraction of treated for ATTE
    p_hat = None
    if score == 'ATTE':
        p_hat = np.mean(d)

    if normalize_ipw:
        m_hat = _normalize_ipw(m_hat, d)

    # compute residuals
    u_hat0 = y - g_hat0
    u_hat1 = None
    if score == 'ATE':
        u_hat1 = y - g_hat1

    psi_a = np.full_like(y, np.nan)
    psi_b = np.full_like(y, np.nan)
    if score == 'ATE':
        psi_b = g_hat1 - g_hat0 \
            + np.divide(np.multiply(d, u_hat1), m_hat) \
            - np.divide(np.multiply(1.0-d, u_hat0), 1.0 - m_hat)
        psi_a = np.full_like(m_hat, -1.0)
    else:
        assert score == 'ATTE'
        psi_b = np.divide(np.multiply(d, u_hat0), p_hat) \
            - np.divide(np.multiply(m_hat, np.multiply(1.0-d, u_hat0)),
                        np.multiply(p_hat, (1.0 - m_hat)))
        psi_a = - np.divide(d, p_hat)

    return psi_a, psi_b


@pytest.fixture(scope='module',
                params=[[LinearRegression(),
                         LogisticRegression(solver='lbfgs', max_iter=250)],
                        [RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
                         RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)]])
def learner(request):
    return request.param


@pytest.fixture(scope='module',
                params=['ATE', 'ATTE'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params=[False, True])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.2, 0.15])
def trimming_threshold(request):
    return request.param


@pytest.fixture(scope='module')
def old_vs_weighted_score_fixture(generate_data_irm, learner, score, normalize_ipw, trimming_threshold):
    n_folds = 2

    # collect data
    (x, y, d) = generate_data_irm
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)

    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

    np.random.seed(3141)
    dml_irm_obj = dml.DoubleMLIRM(obj_dml_data,
                                  ml_g, ml_m,
                                  n_folds,
                                  score=score,
                                  normalize_ipw=normalize_ipw,
                                  trimming_threshold=trimming_threshold)
    dml_irm_obj.fit()

    # old score
    psi_a_old, psi_b_old = old_score_elements(
        y=y,
        d=d,
        g_hat0=np.squeeze(dml_irm_obj.predictions['ml_g0']),
        g_hat1=np.squeeze(dml_irm_obj.predictions['ml_g1']),
        m_hat=np.squeeze(dml_irm_obj.predictions['ml_m']),
        score=score,
        normalize_ipw=normalize_ipw
    )

    old_coef = -np.mean(psi_b_old) / np.mean(psi_a_old)

    result_dict = {
        'psi_a': np.squeeze(dml_irm_obj.psi_elements['psi_a']),
        'psi_b': np.squeeze(dml_irm_obj.psi_elements['psi_b']),
        'psi_a_old': psi_a_old,
        'psi_b_old': psi_b_old,
        'coef': np.squeeze(dml_irm_obj.coef),
        'old_coef': old_coef,
    }
    return result_dict


@pytest.mark.ci
def test_irm_old_vs_weighted_score_psi_b(old_vs_weighted_score_fixture):
    assert np.allclose(old_vs_weighted_score_fixture['psi_b'],
                       old_vs_weighted_score_fixture['psi_b_old'])


@pytest.mark.ci
def test_irm_old_vs_weighted_score_psi_a(old_vs_weighted_score_fixture):
    assert np.allclose(old_vs_weighted_score_fixture['psi_a'],
                       old_vs_weighted_score_fixture['psi_a_old'])


@pytest.mark.ci
def test_irm_old_vs_weighted_coef(old_vs_weighted_score_fixture):
    assert np.allclose(old_vs_weighted_score_fixture['coef'],
                       old_vs_weighted_score_fixture['old_coef'])
