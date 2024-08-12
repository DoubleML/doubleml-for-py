import pytest
import numpy as np
import pandas as pd
import math

from sklearn.linear_model import LinearRegression, LogisticRegression
from doubleml import DoubleMLAPO, DoubleMLData
from doubleml.datasets import make_irm_data_discrete_treatments
from doubleml.utils import DMLDummyRegressor, DMLDummyClassifier

from ...tests._utils import draw_smpls


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
def doubleml_apo_ext_fixture(n_rep, set_ml_m_ext, set_ml_g_ext):

    score = "APO"
    treatment_level = 0
    ext_predictions = {"d": {}}

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
        "treatment_level": treatment_level,
        "n_rep": n_rep,
        "draw_sample_splitting": False
    }

    dml_obj = DoubleMLAPO(ml_g=LinearRegression(), ml_m=LogisticRegression(), **kwargs)
    dml_obj.set_sample_splitting(all_smpls=all_smpls)

    np.random.seed(3141)
    dml_obj.fit(store_predictions=True)

    if set_ml_m_ext:
        ext_predictions["d"]["ml_m"] = dml_obj.predictions["ml_m"][:, :, 0]
        ml_m = DMLDummyClassifier()
    else:
        ml_m = LogisticRegression(random_state=42)

    if set_ml_g_ext:
        ext_predictions["d"]["ml_g0"] = dml_obj.predictions["ml_g0"][:, :, 0]
        ext_predictions["d"]["ml_g1"] = dml_obj.predictions["ml_g1"][:, :, 0]
        ml_g = DMLDummyRegressor()
    else:
        ml_g = LinearRegression()

    dml_obj_ext = DoubleMLAPO(ml_g=ml_g, ml_m=ml_m, **kwargs)
    dml_obj_ext.set_sample_splitting(all_smpls=all_smpls)

    np.random.seed(3141)
    dml_obj_ext.fit(external_predictions=ext_predictions)

    res_dict = {
        "coef_normal": dml_obj.coef[0],
        "coef_ext": dml_obj_ext.coef[0]
    }

    return res_dict


@pytest.mark.ci
def test_doubleml_apo_ext_coef(doubleml_apo_ext_fixture):
    assert math.isclose(
        doubleml_apo_ext_fixture["coef_normal"],
        doubleml_apo_ext_fixture["coef_ext"],
        rel_tol=1e-9,
        abs_tol=1e-4
    )
