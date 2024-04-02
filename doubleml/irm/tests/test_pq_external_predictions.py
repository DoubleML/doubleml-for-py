import numpy as np
import pytest
import math
from sklearn.linear_model import LogisticRegression
from doubleml import DoubleMLPQ, DoubleMLData
from doubleml.datasets import make_irm_data
from doubleml.utils import DMLDummyClassifier
from ...tests._utils import draw_smpls


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_m_ext(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def set_ml_g_ext(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_pq_fixture(n_rep, normalize_ipw, set_ml_m_ext, set_ml_g_ext):
    ext_predictions = {"d": {}}
    np.random.seed(3141)
    data = make_irm_data(theta=1, n_obs=500, dim_x=5, return_type="DataFrame")

    dml_data = DoubleMLData(data, "y", "d")
    all_smpls = draw_smpls(len(dml_data.y), 5, n_rep=n_rep, groups=None)

    kwargs = {
        "obj_dml_data": dml_data,
        "score": "PQ",
        "n_rep": n_rep,
        "normalize_ipw": normalize_ipw,
        "draw_sample_splitting": False,
    }

    ml_m = LogisticRegression(random_state=42)
    ml_g = LogisticRegression(random_state=42)

    dml_pq = DoubleMLPQ(ml_g=ml_g, ml_m=ml_m, **kwargs)
    dml_pq.set_sample_splitting(all_smpls)
    np.random.seed(3141)
    dml_pq.fit(store_predictions=True)

    if set_ml_m_ext:
        ext_predictions["d"]["ml_m"] = dml_pq.predictions["ml_m"][:, :, 0]
        ml_m = DMLDummyClassifier()
    else:
        ml_m = LogisticRegression(random_state=42)

    if set_ml_g_ext:
        ext_predictions["d"]["ml_g"] = dml_pq.predictions["ml_g"][:, :, 0]
        ml_g = DMLDummyClassifier()
    else:
        ml_g = LogisticRegression(random_state=42)

    dml_pq_ext = DoubleMLPQ(ml_g=ml_g, ml_m=ml_m, **kwargs)
    dml_pq_ext.set_sample_splitting(all_smpls)

    np.random.seed(3141)
    dml_pq_ext.fit(external_predictions=ext_predictions)

    if set_ml_m_ext and not set_ml_g_ext:
        # adjust tolerance for the case that ml_m is set to external predictions
        # because no preliminary results are available for ml_m, the model use the (external) final predictions for ml_m
        tol_rel = 0.1
        tol_abs = 0.1
    else:
        tol_rel = 1e-9
        tol_abs = 1e-4

    res_dict = {"coef_normal": dml_pq.coef[0], "coef_ext": dml_pq_ext.coef[0], "tol_rel": tol_rel, "tol_abs": tol_abs}

    return res_dict


@pytest.mark.ci
def test_doubleml_pq_coef(doubleml_pq_fixture):
    assert math.isclose(
        doubleml_pq_fixture["coef_normal"],
        doubleml_pq_fixture["coef_ext"],
        rel_tol=doubleml_pq_fixture["tol_rel"],
        abs_tol=doubleml_pq_fixture["tol_abs"],
    )
