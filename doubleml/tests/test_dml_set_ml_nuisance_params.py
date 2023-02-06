import pytest
import numpy as np

from doubleml import DoubleMLPLR, DoubleMLIRM, DoubleMLIIVM, DoubleMLPLIV, DoubleMLCVAR, DoubleMLPQ, DoubleMLLPQ, DoubleMLQTE
from doubleml.datasets import make_plr_CCDDHNR2018, make_irm_data, make_pliv_CHS2015, make_iivm_data

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# set default and test values
n_est_default = 100
n_est_test = 5
n_folds = 2
test_values = [[{'n_estimators': 5}, {'n_estimators': 5}]]

np.random.seed(3141)
dml_data_plr = make_plr_CCDDHNR2018(n_obs=100)
dml_data_pliv = make_pliv_CHS2015(n_obs=100, dim_z=1)
dml_data_irm = make_irm_data(n_obs=500)
dml_data_iivm = make_iivm_data(n_obs=1000)

# linear models
dml_plr = DoubleMLPLR(dml_data_plr, RandomForestRegressor(), RandomForestRegressor(), n_folds=n_folds)
dml_pliv = DoubleMLPLIV(dml_data_pliv,RandomForestRegressor(), RandomForestRegressor(), RandomForestRegressor(), n_folds=n_folds)
dml_irm = DoubleMLIRM(dml_data_irm, RandomForestRegressor(),RandomForestClassifier(), n_folds=n_folds)
dml_iivm = DoubleMLIIVM(dml_data_iivm, RandomForestRegressor(), RandomForestClassifier(), RandomForestClassifier(), n_folds=n_folds)


dml_plr.set_ml_nuisance_params('ml_l', 'd', {'n_estimators': n_est_test})
dml_pliv.set_ml_nuisance_params('ml_l', 'd', {'n_estimators': n_est_test})
dml_irm.set_ml_nuisance_params('ml_g0', 'd', {'n_estimators': n_est_test})
dml_iivm.set_ml_nuisance_params('ml_g0', 'd', {'n_estimators': n_est_test})

dml_plr.fit(store_models=True)
dml_pliv.fit(store_models=True)
dml_irm.fit(store_models=True)
dml_iivm.fit(store_models=True)

# nonlinear models
dml_pq = DoubleMLPQ(dml_data_irm, ml_g=RandomForestClassifier(), ml_m=RandomForestClassifier(), n_folds=n_folds)

dml_pq.set_ml_nuisance_params('ml_g', 'd', {'n_estimators': n_est_test})

dml_pq.fit(store_models=True)



@pytest.mark.ci
def test_plr_params():
    assert dml_plr.params['ml_l']['d'] == test_values
    assert dml_plr.params['ml_m']['d'][0] is None

    param_list_1 =[dml_plr.models['ml_l']['d'][0][fold].n_estimators for fold in range(n_folds)]
    assert all(param == n_est_test for param in param_list_1)
    param_list_2 =[dml_plr.models['ml_m']['d'][0][fold].n_estimators for fold in range(n_folds)]
    assert all(param == n_est_default for param in param_list_2)

@pytest.mark.ci
def test_pliv_params():
    assert dml_pliv.params['ml_l']['d'] == test_values
    assert dml_pliv.params['ml_m']['d'][0] is None

    param_list_1 =[dml_pliv.models['ml_l']['d'][0][fold].n_estimators for fold in range(n_folds)]
    assert all(param == n_est_test for param in param_list_1)
    param_list_2 =[dml_pliv.models['ml_m']['d'][0][fold].n_estimators for fold in range(n_folds)]
    assert all(param == n_est_default for param in param_list_2)


@pytest.mark.ci
def test_irm_params():
    assert dml_irm.params['ml_g0']['d'] == test_values
    assert dml_irm.params['ml_g1']['d'][0] is None

    param_list_1 =[dml_irm.models['ml_g0']['d'][0][fold].n_estimators for fold in range(n_folds)]
    assert all(param == n_est_test for param in param_list_1)
    param_list_2 =[dml_irm.models['ml_g1']['d'][0][fold].n_estimators for fold in range(n_folds)]
    assert all(param == n_est_default for param in param_list_2)


@pytest.mark.ci
def test_iivm_params():
    assert dml_iivm.params['ml_g0']['d'] == test_values
    assert dml_iivm.params['ml_g1']['d'][0] is None

    param_list_1 =[dml_iivm.models['ml_g0']['d'][0][fold].n_estimators for fold in range(n_folds)]
    assert all(param == n_est_test for param in param_list_1)
    param_list_2 =[dml_iivm.models['ml_g1']['d'][0][fold].n_estimators for fold in range(n_folds)]
    assert all(param == n_est_default for param in param_list_2)


@pytest.mark.ci
def test_pq_params():
    assert dml_pq.params['ml_g']['d'] == test_values
    assert dml_pq.params['ml_m']['d'][0] is None

    param_list_1 =[dml_pq.models['ml_g']['d'][0][fold].n_estimators for fold in range(n_folds)]
    assert all(param == n_est_test for param in param_list_1)
    param_list_2 =[dml_pq.models['ml_m']['d'][0][fold].n_estimators for fold in range(n_folds)]
    assert all(param == n_est_default for param in param_list_2)
