import pytest
import numpy as np

from doubleml import DoubleMLPLR, DoubleMLIRM, DoubleMLIIVM, DoubleMLPLIV
from doubleml.datasets import make_plr_CCDDHNR2018, make_irm_data, make_pliv_CHS2015, make_iivm_data

from sklearn.linear_model import Lasso, LogisticRegression

np.random.seed(3141)
dml_data_plr = make_plr_CCDDHNR2018(n_obs=100)
dml_data_pliv = make_pliv_CHS2015(n_obs=100, dim_z=1)
dml_data_irm = make_irm_data(n_obs=100)
dml_data_iivm = make_iivm_data(n_obs=100)

dml_plr = DoubleMLPLR(dml_data_plr, Lasso(), Lasso())
dml_plr.fit()
dml_plr_iv_type = DoubleMLPLR(dml_data_plr, Lasso(), Lasso(), Lasso(), score='IV-type')
dml_plr_iv_type.fit()
dml_pliv = DoubleMLPLIV(dml_data_pliv, Lasso(), Lasso(), Lasso())
dml_pliv.fit()
dml_irm = DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression(), normalize_ipw=False)
dml_irm.fit()
dml_iivm = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression(), normalize_ipw=False)
dml_iivm.fit()

# fit models with callable scores
plr_score = dml_plr._score_elements
dml_plr_callable_score = DoubleMLPLR(dml_data_plr, Lasso(), Lasso(),
                                     score=plr_score, draw_sample_splitting=False)
dml_plr_callable_score.set_sample_splitting(dml_plr.smpls)
dml_plr_callable_score.fit(store_predictions=True)

plr_iv_type_score = dml_plr_iv_type._score_elements
dml_plr_iv_type_callable_score = DoubleMLPLR(dml_data_plr, Lasso(), Lasso(), Lasso(),
                                             score=plr_iv_type_score, draw_sample_splitting=False)
dml_plr_iv_type_callable_score.set_sample_splitting(dml_plr_iv_type.smpls)
dml_plr_iv_type_callable_score.fit(store_predictions=True)

irm_score = dml_irm._score_elements
dml_irm_callable_score = DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression(),
                                     score=irm_score, draw_sample_splitting=False, normalize_ipw=False)
dml_irm_callable_score.set_sample_splitting(dml_irm.smpls)
dml_irm_callable_score.fit(store_predictions=True)

iivm_score = dml_iivm._score_elements
dml_iivm_callable_score = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression(),
                                       score=iivm_score, draw_sample_splitting=False, normalize_ipw=False)
dml_iivm_callable_score.set_sample_splitting(dml_iivm.smpls)
dml_iivm_callable_score.fit(store_predictions=True)


def non_orth_score_w_g(y, d, l_hat, m_hat, g_hat, smpls):
    u_hat = y - g_hat
    psi_a = -np.multiply(d, d)
    psi_b = np.multiply(d, u_hat)
    return psi_a, psi_b


def non_orth_score_w_l(y, d, l_hat, m_hat, g_hat, smpls):
    p_a = -np.multiply(d - m_hat, d - m_hat)
    p_b = np.multiply(d - m_hat, y - l_hat)
    theta_initial = -np.nanmean(p_b) / np.nanmean(p_a)
    g_hat = l_hat - theta_initial * np.multiply(d, m_hat)

    u_hat = y - g_hat
    psi_a = -np.multiply(d, d)
    psi_b = np.multiply(d, u_hat)
    return psi_a, psi_b


dml_plr_non_orth_score_w_g = DoubleMLPLR(dml_data_plr, Lasso(), Lasso(), Lasso(),
                                         score=non_orth_score_w_g)
dml_plr_non_orth_score_w_g.fit(store_predictions=True)

dml_plr_non_orth_score_w_l = DoubleMLPLR(dml_data_plr, Lasso(), Lasso(),
                                         score=non_orth_score_w_l)
dml_plr_non_orth_score_w_l.fit(store_predictions=True)


@pytest.mark.ci
@pytest.mark.parametrize('dml_obj',
                         [dml_plr, dml_plr_iv_type, dml_pliv, dml_irm, dml_iivm])
def test_linear_score(dml_obj):
    assert np.allclose(dml_obj.psi,
                       dml_obj.psi_elements['psi_a'] * dml_obj.coef + dml_obj.psi_elements['psi_b'],
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_plr_callable_vs_str_score():
    assert np.allclose(dml_plr.psi,
                       dml_plr_callable_score.psi,
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_plr.coef,
                       dml_plr_callable_score.coef,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_plr_callable_vs_pred_export():
    preds = dml_plr_callable_score.predictions
    l_hat = preds['ml_l'].squeeze()
    m_hat = preds['ml_m'].squeeze()
    g_hat = None
    psi_a, psi_b = plr_score(dml_data_plr.y, dml_data_plr.d,
                             l_hat, m_hat, g_hat,
                             dml_plr_callable_score.smpls[0])
    assert np.allclose(dml_plr.psi_elements['psi_a'].squeeze(),
                       psi_a,
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_plr.psi_elements['psi_b'].squeeze(),
                       psi_b,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_plr_iv_type_callable_vs_str_score():
    assert np.allclose(dml_plr_iv_type.psi,
                       dml_plr_iv_type_callable_score.psi,
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_plr_iv_type.coef,
                       dml_plr_iv_type_callable_score.coef,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_plr_iv_type_callable_vs_pred_export():
    preds = dml_plr_iv_type_callable_score.predictions
    l_hat = preds['ml_l'].squeeze()
    m_hat = preds['ml_m'].squeeze()
    g_hat = preds['ml_g'].squeeze()
    psi_a, psi_b = plr_iv_type_score(dml_data_plr.y, dml_data_plr.d,
                                     l_hat, m_hat, g_hat,
                                     dml_plr_iv_type_callable_score.smpls[0])
    assert np.allclose(dml_plr_iv_type.psi_elements['psi_a'].squeeze(),
                       psi_a,
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_plr_iv_type.psi_elements['psi_b'].squeeze(),
                       psi_b,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_plr_non_orth_score_w_g_callable_vs_pred_export():
    preds = dml_plr_non_orth_score_w_g.predictions
    l_hat = preds['ml_l'].squeeze()
    m_hat = preds['ml_m'].squeeze()
    g_hat = preds['ml_g'].squeeze()
    psi_a, psi_b = non_orth_score_w_g(dml_data_plr.y, dml_data_plr.d,
                                      l_hat, m_hat, g_hat,
                                      dml_plr_non_orth_score_w_g.smpls[0])
    assert np.allclose(dml_plr_non_orth_score_w_g.psi_elements['psi_a'].squeeze(),
                       psi_a,
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_plr_non_orth_score_w_g.psi_elements['psi_b'].squeeze(),
                       psi_b,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_plr_non_orth_score_w_l_callable_vs_pred_export():
    preds = dml_plr_non_orth_score_w_l.predictions
    l_hat = preds['ml_l'].squeeze()
    m_hat = preds['ml_m'].squeeze()
    g_hat = None
    psi_a, psi_b = non_orth_score_w_l(dml_data_plr.y, dml_data_plr.d,
                                      l_hat, m_hat, g_hat,
                                      dml_plr_non_orth_score_w_l.smpls[0])
    assert np.allclose(dml_plr_non_orth_score_w_l.psi_elements['psi_a'].squeeze(),
                       psi_a,
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_plr_non_orth_score_w_l.psi_elements['psi_b'].squeeze(),
                       psi_b,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_irm_callable_vs_str_score():
    assert np.allclose(dml_irm.psi,
                       dml_irm_callable_score.psi,
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_irm.coef,
                       dml_irm_callable_score.coef,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_irm_callable_vs_pred_export():
    preds = dml_irm_callable_score.predictions
    g_hat0 = preds['ml_g0'].squeeze()
    g_hat1 = preds['ml_g1'].squeeze()
    m_hat = preds['ml_m'].squeeze()
    psi_a, psi_b = irm_score(dml_data_irm.y, dml_data_irm.d,
                             g_hat0, g_hat1, m_hat,
                             dml_irm_callable_score.smpls[0])
    assert np.allclose(dml_irm.psi_elements['psi_a'].squeeze(),
                       psi_a,
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_irm.psi_elements['psi_b'].squeeze(),
                       psi_b,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_iivm_callable_vs_str_score():
    assert np.allclose(dml_iivm.psi,
                       dml_iivm_callable_score.psi,
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_iivm.coef,
                       dml_iivm_callable_score.coef,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_iivm_callable_vs_pred_export():
    preds = dml_iivm_callable_score.predictions
    g_hat0 = preds['ml_g0'].squeeze()
    g_hat1 = preds['ml_g1'].squeeze()
    m_hat = preds['ml_m'].squeeze()
    r_hat0 = preds['ml_r0'].squeeze()
    r_hat1 = preds['ml_r1'].squeeze()
    psi_a, psi_b = iivm_score(dml_data_iivm.y, dml_data_iivm.z.squeeze(), dml_data_iivm.d,
                              g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                              dml_iivm_callable_score.smpls[0])
    assert np.allclose(dml_iivm.psi_elements['psi_a'].squeeze(),
                       psi_a,
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_iivm.psi_elements['psi_b'].squeeze(),
                       psi_b,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_pliv_callable_vs_str_score():
    pliv_score = dml_pliv._score_elements
    dml_pliv_callable_score = DoubleMLPLIV(dml_data_pliv, Lasso(), Lasso(), Lasso(),
                                           score=pliv_score, draw_sample_splitting=False)
    dml_pliv_callable_score.set_sample_splitting(dml_pliv.smpls)
    dml_pliv_callable_score.fit()
    assert np.allclose(dml_pliv.psi,
                       dml_pliv_callable_score.psi,
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_pliv.coef,
                       dml_pliv_callable_score.coef,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_pliv_callable_not_implemented():
    np.random.seed(3141)
    dml_data_pliv_2z = make_pliv_CHS2015(n_obs=100, dim_z=2)
    pliv_score = dml_pliv._score_elements

    dml_pliv_callable_score = DoubleMLPLIV._partialX(dml_data_pliv_2z, Lasso(), Lasso(), Lasso(),
                                                     score=pliv_score)
    msg = 'Callable score not implemented for DoubleMLPLIV.partialX with several instruments.'
    with pytest.raises(NotImplementedError, match=msg):
        dml_pliv_callable_score.fit()

    dml_pliv_callable_score = DoubleMLPLIV._partialZ(dml_data_pliv_2z, Lasso(),
                                                     score=pliv_score)
    msg = 'Callable score not implemented for DoubleMLPLIV.partialZ.'
    with pytest.raises(NotImplementedError, match=msg):
        dml_pliv_callable_score.fit()

    dml_pliv_callable_score = DoubleMLPLIV._partialXZ(dml_data_pliv_2z, Lasso(), Lasso(), Lasso(),
                                                      score=pliv_score)
    msg = 'Callable score not implemented for DoubleMLPLIV.partialXZ.'
    with pytest.raises(NotImplementedError, match=msg):
        dml_pliv_callable_score.fit()
