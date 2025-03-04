import math

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.did.datasets import make_did_CS2021
from doubleml.utils import DMLDummyClassifier, DMLDummyRegressor


@pytest.fixture(scope="module", params=["observational", "experimental"])
def did_score(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_m_ext(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_g_ext(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_did_multi_ext_fixture(did_score, n_rep, set_ml_m_ext, set_ml_g_ext):
    n_obs = 500
    n_folds = 5
    dgp = 1
    ml_g = LinearRegression()
    ml_m = LogisticRegression(random_state=42)

    # collect data
    df = make_did_CS2021(n_obs=n_obs, dgp_type=dgp, time_type="float")
    dml_panel_data = dml.data.DoubleMLPanelData(
        df, y_col="y", d_cols="d", id_col="id", t_col="t", x_cols=["Z1", "Z2", "Z3", "Z4"]
    )

    dml_args = {
        "obj_dml_data": dml_panel_data,
        "gt_combinations": [(2, 0, 1)],
        "score": did_score,
        "n_rep": n_rep,
        "n_folds": n_folds,
    }

    np.random.seed(3141)
    dml_obj = dml.did.DoubleMLDIDMulti(
        ml_g=ml_g,
        ml_m=ml_m,
        **dml_args,
    )
    np.random.seed(3141)
    dml_obj.fit()

    ext_pred_dict = {gt_combination: {} for gt_combination in dml_obj.gt_labels}
    if set_ml_m_ext and did_score == "observational":
        for i_gt_combination, gt_label in enumerate(dml_obj.gt_labels):
            ext_pred_dict[gt_label]["ml_m"] = dml_obj.modellist[i_gt_combination].predictions["ml_m"][:, :, 0]
        ml_m_ext = DMLDummyClassifier()
    else:
        ml_m_ext = ml_m

    if set_ml_g_ext:
        for i_gt_combination, gt_label in enumerate(dml_obj.gt_labels):
            ext_pred_dict[gt_label]["ml_g0"] = dml_obj.modellist[i_gt_combination].predictions["ml_g0"][:, :, 0]
            ext_pred_dict[gt_label]["ml_g1"] = dml_obj.modellist[i_gt_combination].predictions["ml_g1"][:, :, 0]
        ml_g_ext = DMLDummyRegressor()
    else:
        ml_g_ext = ml_g

    np.random.seed(3141)
    dml_obj_ext = dml.did.DoubleMLDIDMulti(
        ml_g=ml_g_ext,
        ml_m=ml_m_ext,
        **dml_args,
    )
    np.random.seed(3141)
    dml_obj_ext.fit(external_predictions=ext_pred_dict)

    res_dict = {
        "coef": dml_obj.coef[0],
        "coef_ext": dml_obj_ext.coef[0],
        "se": dml_obj.se[0],
        "se_ext": dml_obj_ext.se[0],
    }

    return res_dict


@pytest.mark.ci
def test_coef(doubleml_did_multi_ext_fixture):
    assert math.isclose(
        doubleml_did_multi_ext_fixture["coef"], doubleml_did_multi_ext_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-3
    )
