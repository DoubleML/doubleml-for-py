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
dml_pliv = DoubleMLPLIV(dml_data_pliv, Lasso(), Lasso(), Lasso())
dml_irm = DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression())
dml_iivm = DoubleMLIIVM(dml_data_iivm, Lasso(), LogisticRegression(), LogisticRegression())


def _assert_resampling_default_settings(dml_obj):
    assert dml_obj.n_folds == 5
    assert dml_obj.n_rep == 1
    assert dml_obj.draw_sample_splitting
    assert dml_obj.apply_cross_fitting


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
    assert dml_irm.trimming_threshold == 1e-12


@pytest.mark.ci
def test_iivm_defaults():
    _assert_resampling_default_settings(dml_iivm)
    assert dml_iivm.score == 'LATE'
    assert dml_iivm.subgroups == {'always_takers': True, 'never_takers': True}
    assert dml_iivm.dml_procedure == 'dml2'
    assert dml_iivm.trimming_rule == 'truncate'
    assert dml_iivm.trimming_threshold == 1e-12
