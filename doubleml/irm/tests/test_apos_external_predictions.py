import pytest
import numpy as np
import pandas as pd
import math

from sklearn.linear_model import LinearRegression, LogisticRegression
from doubleml import DoubleMLAPOS, DoubleMLData
from doubleml.datasets import make_irm_data_discrete_treatments
from doubleml.utils import DMLDummyRegressor, DMLDummyClassifier

from ...tests._utils import draw_smpls


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module", params=[[0, 1, 2, 3], [0, 1]])
def treatment_levels(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_m_ext(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_g_ext(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_apos_ext_fixture(n_rep, treatment_levels, set_ml_m_ext, set_ml_g_ext):
    score = "APO"
    ext_predictions = {
        treatment_level: {} for treatment_level in treatment_levels
    }

    np.random.seed(3141)
    n_obs = 500
    data_apo = make_irm_data_discrete_treatments(n_obs=n_obs)
    df_apo = pd.DataFrame(
        np.column_stack((data_apo['y'], data_apo['d'], data_apo['x'])),
        columns=['y', 'd'] + ['x' + str(i) for i in range(data_apo['x'].shape[1])]
    )

    dml_data = DoubleMLData(df_apo, 'y', 'd')
    d = data_apo['d']
    all_smpls = draw_smpls(n_obs, n_folds=5, n_rep=n_rep, groups=d)

    kwargs = {
        "obj_dml_data": dml_data,
        "score": score,
        "treatment_levels": treatment_levels,
        "n_rep": n_rep,
        "draw_sample_splitting": False
    }

    dml_obj = DoubleMLAPOS(ml_g=LinearRegression(), ml_m=LogisticRegression(), **kwargs)
    dml_obj.set_sample_splitting(all_smpls=all_smpls)

    np.random.seed(3141)
    dml_obj.fit(store_predictions=True)

    if set_ml_m_ext:
        for i_treatment_level, treatment_level in enumerate(treatment_levels):
            ext_predictions[treatment_level]["ml_m"] = dml_obj.modellist[i_treatment_level].predictions["ml_m"][:, :, 0]
        ml_m = DMLDummyClassifier()
    else:
        ml_m = LogisticRegression(random_state=42)

    if set_ml_g_ext:
        for i_treatment_level, treatment_level in enumerate(treatment_levels):
            ext_predictions[treatment_level]["ml_g0"] = dml_obj.modellist[i_treatment_level].predictions["ml_g0"][:, :, 0]
            ext_predictions[treatment_level]["ml_g1"] = dml_obj.modellist[i_treatment_level].predictions["ml_g1"][:, :, 0]
        ml_g = DMLDummyRegressor()
    else:
        ml_g = LinearRegression()

    dml_obj_ext = DoubleMLAPOS(ml_g=ml_g, ml_m=ml_m, **kwargs)
    dml_obj_ext.set_sample_splitting(all_smpls=all_smpls)

    np.random.seed(3141)
    dml_obj_ext.fit(external_predictions=ext_predictions)

    res_dict = {
        "coef_normal": dml_obj.coef[0],
        "coef_ext": dml_obj_ext.coef[0],
        "dml_obj": dml_obj,
        "dml_obj_ext": dml_obj_ext,
        "treatment_levels": treatment_levels
    }

    return res_dict


@pytest.mark.ci
def test_doubleml_apos_ext_coef(doubleml_apos_ext_fixture):
    assert math.isclose(
        doubleml_apos_ext_fixture["coef_normal"],
        doubleml_apos_ext_fixture["coef_ext"],
        rel_tol=1e-9,
        abs_tol=1e-4
    )


@pytest.mark.ci
def test_doubleml_apos_ext_pred_nuisance(doubleml_apos_ext_fixture):
    for i_level, _ in enumerate(doubleml_apos_ext_fixture["treatment_levels"]):
        for nuisance_key in ["ml_g0", "ml_g1", "ml_m"]:
            assert np.allclose(
                doubleml_apos_ext_fixture["dml_obj"].modellist[i_level].nuisance_loss[nuisance_key],
                doubleml_apos_ext_fixture["dml_obj_ext"].modellist[i_level].nuisance_loss[nuisance_key],
                rtol=1e-9,
                atol=1e-4
            )
