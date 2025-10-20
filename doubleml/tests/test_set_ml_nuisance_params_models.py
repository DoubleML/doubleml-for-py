import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from doubleml import DoubleMLAPO, DoubleMLCVAR, DoubleMLIIVM, DoubleMLIRM, DoubleMLLPQ, DoubleMLPLIV, DoubleMLPLR, DoubleMLPQ
from doubleml.data import DoubleMLPanelData
from doubleml.did import DoubleMLDIDBinary, DoubleMLDIDCSBinary
from doubleml.did.datasets import make_did_CS2021
from doubleml.irm.datasets import make_iivm_data, make_irm_data
from doubleml.plm.datasets import make_pliv_CHS2015, make_plr_CCDDHNR2018

# set default and test values
n_est_default = 100
n_est_test = 5
n_folds = 2
test_values = [[{"n_estimators": 5}, {"n_estimators": 5}]]

np.random.seed(3141)
dml_data_plr = make_plr_CCDDHNR2018(n_obs=100)
dml_data_pliv = make_pliv_CHS2015(n_obs=100, dim_z=1)
dml_data_irm = make_irm_data(n_obs=1000)
dml_data_iivm = make_iivm_data(n_obs=2000)

# Create DID data
df_did = make_did_CS2021(n_obs=500, dgp_type=1, n_pre_treat_periods=2, n_periods=4, time_type="float")
dml_data_did = DoubleMLPanelData(df_did, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"])

reg_learner = RandomForestRegressor(max_depth=2)
class_learner = RandomForestClassifier(max_depth=2)

# linear models
dml_plr = DoubleMLPLR(dml_data_plr, reg_learner, reg_learner, n_folds=n_folds)
dml_pliv = DoubleMLPLIV(dml_data_pliv, reg_learner, reg_learner, reg_learner, n_folds=n_folds)
dml_irm = DoubleMLIRM(dml_data_irm, reg_learner, class_learner, n_folds=n_folds)
dml_iivm = DoubleMLIIVM(dml_data_iivm, reg_learner, class_learner, class_learner, n_folds=n_folds)
dml_cvar = DoubleMLCVAR(dml_data_irm, ml_g=reg_learner, ml_m=class_learner, n_folds=n_folds)
dml_apo = DoubleMLAPO(dml_data_irm, ml_g=reg_learner, ml_m=class_learner, n_folds=n_folds, treatment_level=1)

dml_plr.set_ml_nuisance_params("ml_l", "d", {"n_estimators": n_est_test})
dml_pliv.set_ml_nuisance_params("ml_l", "d", {"n_estimators": n_est_test})
dml_irm.set_ml_nuisance_params("ml_g0", "d", {"n_estimators": n_est_test})
dml_iivm.set_ml_nuisance_params("ml_g0", "d", {"n_estimators": n_est_test})
dml_cvar.set_ml_nuisance_params("ml_g", "d", {"n_estimators": n_est_test})
dml_apo.set_ml_nuisance_params("ml_g_d_lvl1", "d", {"n_estimators": n_est_test})

dml_plr.fit(store_models=True)
dml_pliv.fit(store_models=True)
dml_irm.fit(store_models=True)
dml_iivm.fit(store_models=True)
dml_cvar.fit(store_models=True)
dml_apo.fit(store_models=True)

# DID models
dml_did_binary = DoubleMLDIDBinary(
    obj_dml_data=dml_data_did,
    ml_g=reg_learner,
    ml_m=class_learner,
    g_value=2,
    t_value_pre=0,
    t_value_eval=1,
    score="observational",
    n_folds=n_folds,
)

dml_did_cs_binary = DoubleMLDIDCSBinary(
    obj_dml_data=dml_data_did,
    ml_g=reg_learner,
    ml_m=class_learner,
    g_value=2,
    t_value_pre=0,
    t_value_eval=1,
    score="observational",
    n_folds=n_folds,
)

dml_did_binary.set_ml_nuisance_params("ml_g0", "d", {"n_estimators": n_est_test})
dml_did_cs_binary.set_ml_nuisance_params("ml_g_d0_t0", "d", {"n_estimators": n_est_test})

dml_did_binary.fit(store_models=True)
dml_did_cs_binary.fit(store_models=True)

# nonlinear models
dml_pq = DoubleMLPQ(dml_data_irm, ml_g=class_learner, ml_m=class_learner, n_folds=n_folds)
dml_lpq = DoubleMLLPQ(dml_data_iivm, ml_g=class_learner, ml_m=class_learner, n_folds=n_folds)

dml_pq.set_ml_nuisance_params("ml_g", "d", {"n_estimators": n_est_test})
dml_lpq.set_ml_nuisance_params("ml_m_z", "d", {"n_estimators": n_est_test})

dml_pq.fit(store_models=True)
dml_lpq.fit(store_models=True)


def _assert_nuisance_params(dml_obj, learner_1, learner_2):
    assert dml_obj.params[learner_1]["d"] == test_values
    assert dml_obj.params[learner_2]["d"][0] is None

    param_list_1 = [dml_obj.models[learner_1]["d"][0][fold].n_estimators for fold in range(n_folds)]
    assert all(param == n_est_test for param in param_list_1)
    param_list_2 = [dml_obj.models[learner_2]["d"][0][fold].n_estimators for fold in range(n_folds)]
    assert all(param == n_est_default for param in param_list_2)


@pytest.mark.ci
def test_plr_params():
    _assert_nuisance_params(dml_plr, "ml_l", "ml_m")


@pytest.mark.ci
def test_pliv_params():
    _assert_nuisance_params(dml_pliv, "ml_l", "ml_m")


@pytest.mark.ci
def test_irm_params():
    _assert_nuisance_params(dml_irm, "ml_g0", "ml_g1")


@pytest.mark.ci
def test_apo_params():
    _assert_nuisance_params(dml_apo, "ml_g_d_lvl1", "ml_m")


@pytest.mark.ci
def test_iivm_params():
    _assert_nuisance_params(dml_iivm, "ml_g0", "ml_g1")


@pytest.mark.ci
def test_cvar_params():
    _assert_nuisance_params(dml_cvar, "ml_g", "ml_m")


@pytest.mark.ci
def test_pq_params():
    _assert_nuisance_params(dml_pq, "ml_g", "ml_m")


@pytest.mark.ci
def test_lpq_params():
    _assert_nuisance_params(dml_lpq, "ml_m_z", "ml_m_d_z0")
    param_list_2 = [dml_lpq.models["ml_m_d_z1"]["d"][0][fold].n_estimators for fold in range(n_folds)]
    assert all(param == n_est_default for param in param_list_2)
    param_list_2 = [dml_lpq.models["ml_g_du_z0"]["d"][0][fold].n_estimators for fold in range(n_folds)]
    assert all(param == n_est_default for param in param_list_2)
    param_list_2 = [dml_lpq.models["ml_g_du_z1"]["d"][0][fold].n_estimators for fold in range(n_folds)]
    assert all(param == n_est_default for param in param_list_2)


@pytest.mark.ci
def test_did_binary_params():
    _assert_nuisance_params(dml_did_binary, "ml_g0", "ml_g1")


@pytest.mark.ci
def test_did_cs_binary_params():
    _assert_nuisance_params(dml_did_cs_binary, "ml_g_d0_t0", "ml_g_d1_t0")
