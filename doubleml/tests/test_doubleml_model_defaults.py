import pytest
import numpy as np

from doubleml import DoubleMLPLR, DoubleMLIRM, DoubleMLIIVM, DoubleMLPLIV, DoubleMLCVAR, DoubleMLPQ, \
    DoubleMLLPQ, DoubleMLQTE, DoubleMLDID, DoubleMLDIDCS
from doubleml.datasets import make_plr_CCDDHNR2018, make_irm_data, make_pliv_CHS2015, make_iivm_data, make_did_SZ2020

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

np.random.seed(3141)
dml_data_plr = make_plr_CCDDHNR2018(n_obs=100)
dml_data_pliv = make_pliv_CHS2015(n_obs=100, dim_z=1)
dml_data_irm = make_irm_data(n_obs=500)
dml_data_iivm = make_iivm_data(n_obs=1000)
dml_data_did = make_did_SZ2020(n_obs=100)
dml_data_did_cs = make_did_SZ2020(n_obs=100, cross_sectional_data=True)

# linear models
dml_plr = DoubleMLPLR(dml_data_plr, Lasso(), Lasso())
dml_pliv = DoubleMLPLIV(dml_data_pliv, Lasso(), Lasso(), Lasso())
dml_irm = DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression())
dml_iivm = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression())
dml_cvar = DoubleMLCVAR(dml_data_irm, ml_g=RandomForestRegressor(), ml_m=RandomForestClassifier())
dml_did = DoubleMLDID(dml_data_did, Lasso(), LogisticRegression())
dml_did_cs = DoubleMLDIDCS(dml_data_did_cs, Lasso(), LogisticRegression())

dml_plr.fit()
dml_pliv.fit()
dml_irm.fit()
dml_iivm.fit()
dml_cvar.fit()
dml_did.fit()
dml_did_cs.fit()

dml_plr.bootstrap()
dml_pliv.bootstrap()
dml_irm.bootstrap()
dml_iivm.bootstrap()
dml_cvar.bootstrap()
dml_did.bootstrap()
dml_did_cs.bootstrap()

# nonlinear models
dml_pq = DoubleMLPQ(dml_data_irm, ml_g=LogisticRegression(), ml_m=LogisticRegression())
dml_lpq = DoubleMLLPQ(dml_data_iivm, ml_g=RandomForestClassifier(), ml_m=RandomForestClassifier())
dml_qte = DoubleMLQTE(dml_data_irm, ml_g=RandomForestClassifier(), ml_m=RandomForestClassifier())

dml_pq.fit()
dml_lpq.fit()
dml_qte.fit()

dml_pq.bootstrap()
dml_lpq.bootstrap()
dml_qte.bootstrap()


def _assert_resampling_default_settings(dml_obj):
    assert dml_obj.n_folds == 5
    assert dml_obj.n_rep == 1
    assert dml_obj.draw_sample_splitting
    assert dml_obj.apply_cross_fitting

    # fit method
    assert dml_obj.predictions is not None
    assert dml_obj.models is None

    # bootstrap method
    assert dml_obj.boot_method == 'normal'
    assert dml_obj.n_rep_boot == 500

    # confint method
    assert dml_obj.confint().equals(dml_obj.confint(joint=False, level=0.95))

    # p_adjust method
    assert dml_obj.p_adjust().equals(dml_obj.p_adjust(method='romano-wolf'))


@pytest.mark.ci
def test_plr_defaults():
    _assert_resampling_default_settings(dml_plr)
    assert dml_plr.score == 'partialling out'
    assert dml_plr.dml_procedure == 'dml2'


@pytest.mark.ci
def test_pliv_defaults():
    _assert_resampling_default_settings(dml_pliv)
    assert dml_pliv.score == 'partialling out'
    assert dml_pliv.dml_procedure == 'dml2'
    assert dml_pliv.partialX
    assert not dml_pliv.partialZ


@pytest.mark.ci
def test_irm_defaults():
    _assert_resampling_default_settings(dml_irm)
    assert dml_irm.score == 'ATE'
    assert dml_irm.dml_procedure == 'dml2'
    assert dml_irm.trimming_rule == 'truncate'
    assert dml_irm.trimming_threshold == 1e-2
    assert not dml_irm.normalize_ipw


@pytest.mark.ci
def test_iivm_defaults():
    _assert_resampling_default_settings(dml_iivm)
    assert dml_iivm.score == 'LATE'
    assert dml_iivm.subgroups == {'always_takers': True, 'never_takers': True}
    assert dml_iivm.dml_procedure == 'dml2'
    assert dml_iivm.trimming_rule == 'truncate'
    assert dml_iivm.trimming_threshold == 1e-2
    assert not dml_iivm.normalize_ipw


@pytest.mark.ci
def test_cvar_defaults():
    _assert_resampling_default_settings(dml_cvar)
    assert dml_cvar.quantile == 0.5
    assert dml_cvar.treatment == 1
    assert dml_cvar.score == 'CVaR'
    assert dml_cvar.dml_procedure == 'dml2'
    assert dml_cvar.trimming_rule == 'truncate'
    assert dml_cvar.trimming_threshold == 1e-2


@pytest.mark.ci
def test_pq_defaults():
    _assert_resampling_default_settings(dml_pq)
    assert dml_pq.quantile == 0.5
    assert dml_pq.treatment == 1
    assert dml_pq.score == 'PQ'
    assert dml_pq.dml_procedure == 'dml2'
    assert dml_pq.trimming_rule == 'truncate'
    assert dml_pq.trimming_threshold == 1e-2
    assert dml_pq.normalize_ipw


@pytest.mark.ci
def test_lpq_defaults():
    _assert_resampling_default_settings(dml_lpq)
    assert dml_lpq.quantile == 0.5
    assert dml_lpq.treatment == 1
    assert dml_lpq.score == 'LPQ'
    assert dml_lpq.dml_procedure == 'dml2'
    assert dml_lpq.trimming_rule == 'truncate'
    assert dml_lpq.trimming_threshold == 1e-2
    assert dml_lpq.normalize_ipw


@pytest.mark.ci
def test_qte_defaults():
    # not fix since its a differen object added in future versions _assert_resampling_default_settings(dml_qte)
    assert dml_qte.quantiles == 0.5
    assert dml_qte.score == 'PQ'
    assert dml_qte.dml_procedure == 'dml2'
    assert dml_qte.trimming_rule == 'truncate'
    assert dml_qte.trimming_threshold == 1e-2
    assert dml_qte.normalize_ipw


@pytest.mark.ci
def test_did_defaults():
    _assert_resampling_default_settings(dml_did)
    assert dml_did.score == 'observational'
    assert dml_did.in_sample_normalization
    assert dml_did.trimming_rule == 'truncate'
    assert dml_did.trimming_threshold == 1e-2


@pytest.mark.ci
def test_did_cs_defaults():
    _assert_resampling_default_settings(dml_did_cs)
    assert dml_did.score == 'observational'
    assert dml_did_cs.in_sample_normalization
    assert dml_did_cs.dml_procedure == 'dml2'
    assert dml_did_cs.trimming_rule == 'truncate'
    assert dml_did_cs.trimming_threshold == 1e-2
