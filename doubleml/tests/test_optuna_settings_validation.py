import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import doubleml as dml
from doubleml.did.datasets import make_did_SZ2020
from doubleml.irm.datasets import make_irm_data
from doubleml.plm.datasets import make_pliv_CHS2015, make_plr_CCDDHNR2018


def _constant_params(_trial):
    return {}


@pytest.mark.ci
def test_optuna_settings_invalid_key_for_irm_raises():
    np.random.seed(2024)
    dml_data = make_irm_data(n_obs=40, dim_x=2)

    ml_g = DecisionTreeRegressor(random_state=11)
    ml_m = DecisionTreeClassifier(random_state=22)
    dml_irm = dml.DoubleMLIRM(dml_data, ml_g, ml_m, n_folds=2, n_rep=1)

    optuna_params = {"ml_g0": _constant_params, "ml_g1": _constant_params, "ml_m": _constant_params}
    invalid_settings = {"ml_l": {"n_trials": 5}}

    with pytest.raises(ValueError, match="ml_l"):
        dml_irm.tune_ml_models(ml_param_space=optuna_params, optuna_settings=invalid_settings)


@pytest.mark.ci
def test_optuna_settings_invalid_key_for_plr_raises():
    np.random.seed(2025)
    dml_data = make_plr_CCDDHNR2018(n_obs=80, dim_x=4)

    ml_l = DecisionTreeRegressor(random_state=33)
    ml_m = DecisionTreeRegressor(random_state=44)
    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, n_rep=1)

    optuna_params = {"ml_l": _constant_params, "ml_m": _constant_params}
    invalid_settings = {"ml_g0": {"n_trials": 5}}

    with pytest.raises(ValueError, match="ml_g0"):
        dml_plr.tune_ml_models(ml_param_space=optuna_params, optuna_settings=invalid_settings)


@pytest.mark.ci
def test_optuna_settings_invalid_key_for_pliv_raises():
    np.random.seed(2026)
    dml_data = make_pliv_CHS2015(n_obs=80, dim_x=4, dim_z=2)

    ml_l = DecisionTreeRegressor(random_state=55)
    ml_m = DecisionTreeRegressor(random_state=66)
    ml_r = DecisionTreeRegressor(random_state=77)
    dml_pliv = dml.DoubleMLPLIV(dml_data, ml_l, ml_m, ml_r, n_folds=2, n_rep=1)

    optuna_params = {
        "ml_l": _constant_params,
        "ml_m_Z1": _constant_params,
        "ml_m_Z2": _constant_params,
        "ml_r": _constant_params,
    }

    invalid_settings = {"ml_g": {"n_trials": 5}}

    with pytest.raises(ValueError, match="ml_g"):
        dml_pliv.tune_ml_models(ml_param_space=optuna_params, optuna_settings=invalid_settings)


@pytest.mark.ci
def test_optuna_settings_invalid_key_for_did_raises():
    np.random.seed(2027)
    dml_data = make_did_SZ2020(n_obs=120, dgp_type=1, return_type="DoubleMLDIDData")

    ml_g = DecisionTreeRegressor(random_state=88)
    ml_m = DecisionTreeClassifier(random_state=99)
    dml_did = dml.DoubleMLDID(dml_data, ml_g, ml_m, score="observational", n_folds=2, n_rep=1)

    optuna_params = {"ml_g0": _constant_params, "ml_g1": _constant_params, "ml_m": _constant_params}
    invalid_settings = {"ml_l": {"n_trials": 5}}

    with pytest.raises(ValueError, match="ml_l"):
        dml_did.tune_ml_models(ml_param_space=optuna_params, optuna_settings=invalid_settings)


@pytest.mark.ci
def test_optuna_params_invalid_key_for_irm_raises():
    np.random.seed(2028)
    dml_data = make_irm_data(n_obs=40, dim_x=2)

    ml_g = DecisionTreeRegressor(random_state=99)
    ml_m = DecisionTreeClassifier(random_state=101)
    dml_irm = dml.DoubleMLIRM(dml_data, ml_g, ml_m, n_folds=2, n_rep=1)

    optuna_params = {"ml_g0": _constant_params, "ml_g1": _constant_params, "ml_m": _constant_params, "ml_l": _constant_params}

    with pytest.raises(ValueError, match="ml_l"):
        dml_irm.tune_ml_models(ml_param_space=optuna_params)


@pytest.mark.ci
def test_optuna_params_invalid_key_for_plr_raises():
    np.random.seed(2029)
    dml_data = make_plr_CCDDHNR2018(n_obs=80, dim_x=4)

    ml_l = DecisionTreeRegressor(random_state=111)
    ml_m = DecisionTreeRegressor(random_state=222)
    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=2, n_rep=1)

    optuna_params = {"ml_l": _constant_params, "ml_m": _constant_params, "ml_g0": _constant_params}

    with pytest.raises(ValueError, match="ml_g0"):
        dml_plr.tune_ml_models(ml_param_space=optuna_params)


@pytest.mark.ci
def test_optuna_params_invalid_key_for_pliv_raises():
    np.random.seed(2030)
    dml_data = make_pliv_CHS2015(n_obs=80, dim_x=4, dim_z=2)

    ml_l = DecisionTreeRegressor(random_state=333)
    ml_m = DecisionTreeRegressor(random_state=444)
    ml_r = DecisionTreeRegressor(random_state=555)
    dml_pliv = dml.DoubleMLPLIV(dml_data, ml_l, ml_m, ml_r, n_folds=2, n_rep=1)

    optuna_params = {"ml_l": _constant_params, "ml_m": _constant_params, "ml_r": _constant_params, "ml_g": _constant_params}

    with pytest.raises(ValueError, match="ml_g"):
        dml_pliv.tune_ml_models(ml_param_space=optuna_params)


@pytest.mark.ci
def test_optuna_params_invalid_key_for_did_raises():
    np.random.seed(2031)
    dml_data = make_did_SZ2020(n_obs=100, dgp_type=1, return_type="DoubleMLDIDData")

    ml_g = DecisionTreeRegressor(random_state=666)
    dml_did = dml.DoubleMLDID(dml_data, ml_g, score="experimental", n_folds=2, n_rep=1)

    optuna_params = {"ml_g0": _constant_params, "ml_g1": _constant_params, "ml_l": _constant_params}

    with pytest.raises(ValueError, match="ml_l"):
        dml_did.tune_ml_models(ml_param_space=optuna_params)
