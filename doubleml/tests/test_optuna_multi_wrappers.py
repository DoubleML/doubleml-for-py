from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.data import DoubleMLPanelData
from doubleml.did.datasets import make_did_CS2021
from doubleml.irm.datasets import make_irm_data, make_irm_data_discrete_treatments
from doubleml.tests._utils_tune_optuna import _basic_optuna_settings, _small_tree_params


def _build_apos_object():
    np.random.seed(3141)
    data = make_irm_data_discrete_treatments(n_obs=40, n_levels=3, random_state=42)
    x = data["x"]
    y = data["y"]
    d = data["d"]
    columns = ["y", "d"] + [f"x{i + 1}" for i in range(x.shape[1])]
    df = pd.DataFrame(np.column_stack((y, d, x)), columns=columns)
    dml_data = dml.DoubleMLData(df, "y", "d")

    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=200)

    return dml.DoubleMLAPOS(
        dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        treatment_levels=[0, 1, 2],
        n_folds=2,
        n_rep=1,
    )


def _build_qte_object():
    np.random.seed(3141)
    dml_data = make_irm_data(n_obs=80, dim_x=5)
    ml = LogisticRegression(max_iter=200)

    return dml.DoubleMLQTE(
        dml_data,
        ml_g=ml,
        ml_m=ml,
        quantiles=[0.25, 0.75],
        n_folds=2,
        n_rep=1,
    )


def _build_did_multi_object():
    np.random.seed(3141)
    df = make_did_CS2021(n_obs=40, n_periods=4, time_type="datetime")
    x_cols = [col for col in df.columns if col.startswith("Z")]
    dml_panel = DoubleMLPanelData(df, y_col="y", d_cols="d", t_col="t", id_col="id", x_cols=x_cols)

    ml_g = LinearRegression()
    ml_m = LogisticRegression(max_iter=200)

    return dml.did.DoubleMLDIDMulti(
        obj_dml_data=dml_panel,
        ml_g=ml_g,
        ml_m=ml_m,
        control_group="never_treated",
        n_folds=2,
        n_rep=1,
        panel=True,
    )


def _collect_param_names(dml_obj):
    if hasattr(dml_obj, "params_names") and dml_obj.params_names:
        return dml_obj.params_names
    learner_dict = getattr(dml_obj, "_learner", None)
    if isinstance(learner_dict, dict) and learner_dict:
        return list(learner_dict.keys())
    return ["ml_g"]


def _make_tune_kwargs(dml_obj, return_tune_res=True):
    optuna_params = {name: _small_tree_params for name in _collect_param_names(dml_obj)}
    return {
        "ml_param_space": optuna_params,
        "cv": 3,
        "set_as_params": False,
        "return_tune_res": return_tune_res,
        "optuna_settings": _basic_optuna_settings({"n_trials": 2}),
        "scoring_methods": None,
    }


@pytest.fixture
def apos_obj():
    return _build_apos_object()


@pytest.fixture
def qte_obj():
    return _build_qte_object()


@pytest.fixture
def did_multi_obj():
    return _build_did_multi_object()


@pytest.mark.ci
def test_doubleml_apos_tune_ml_models_collects_results(apos_obj):
    dml_obj = apos_obj
    mocks = []
    expected_payload = []

    for idx in range(dml_obj.n_treatment_levels):
        mock_model = MagicMock()
        payload = {"params": f"level-{idx}"}
        mock_model.tune_ml_models.return_value = [payload]
        mocks.append(mock_model)
        expected_payload.append(payload)

    dml_obj._modellist = mocks

    tune_kwargs = _make_tune_kwargs(dml_obj)

    res = dml_obj.tune_ml_models(**tune_kwargs)
    assert res == expected_payload
    for mock in mocks:
        mock.tune_ml_models.assert_called_once_with(**tune_kwargs)

    for mock in mocks:
        mock.reset_mock()
    tune_kwargs_nores = _make_tune_kwargs(dml_obj, return_tune_res=False)

    assert dml_obj.tune_ml_models(**tune_kwargs_nores) is dml_obj
    for mock in mocks:
        mock.tune_ml_models.assert_called_once_with(**tune_kwargs_nores)


@pytest.mark.ci
def test_doubleml_qte_tune_ml_models_returns_quantile_results(qte_obj):
    dml_obj = qte_obj
    modellist_0 = []
    modellist_1 = []
    expected_payload = []

    for idx in range(dml_obj.n_quantiles):
        mock_0 = MagicMock()
        mock_1 = MagicMock()
        payload_0 = {"params": f"quantile-{idx}-treatment-0"}
        payload_1 = {"params": f"quantile-{idx}-treatment-1"}
        mock_0.tune_ml_models.return_value = [payload_0]
        mock_1.tune_ml_models.return_value = [payload_1]
        modellist_0.append(mock_0)
        modellist_1.append(mock_1)
        expected_payload.append({"treatment_0": payload_0, "treatment_1": payload_1})

    dml_obj._modellist_0 = modellist_0
    dml_obj._modellist_1 = modellist_1

    tune_kwargs = _make_tune_kwargs(dml_obj)

    res = dml_obj.tune_ml_models(**tune_kwargs)
    assert res == expected_payload
    for mock in modellist_0 + modellist_1:
        mock.tune_ml_models.assert_called_once_with(**tune_kwargs)

    for mock in modellist_0 + modellist_1:
        mock.reset_mock()
    tune_kwargs_nores = _make_tune_kwargs(dml_obj, return_tune_res=False)

    assert dml_obj.tune_ml_models(**tune_kwargs_nores) is dml_obj
    for mock in modellist_0 + modellist_1:
        mock.tune_ml_models.assert_called_once_with(**tune_kwargs_nores)


@pytest.mark.ci
def test_doubleml_did_multi_tune_ml_models_handles_all_group_time_models(did_multi_obj):
    dml_obj = did_multi_obj
    mocks = []
    expected_payload = []

    for idx in range(len(dml_obj.modellist)):
        mock_model = MagicMock()
        payload = {"params": f"gt-{idx}"}
        mock_model.tune_ml_models.return_value = [payload]
        mocks.append(mock_model)
        expected_payload.append(payload)

    dml_obj._modellist = mocks

    tune_kwargs = _make_tune_kwargs(dml_obj)

    res = dml_obj.tune_ml_models(**tune_kwargs)
    assert res == expected_payload
    for mock in mocks:
        mock.tune_ml_models.assert_called_once_with(**tune_kwargs)

    for mock in mocks:
        mock.reset_mock()
    tune_kwargs_nores = _make_tune_kwargs(dml_obj, return_tune_res=False)

    assert dml_obj.tune_ml_models(**tune_kwargs_nores) is dml_obj
    for mock in mocks:
        mock.tune_ml_models.assert_called_once_with(**tune_kwargs_nores)
