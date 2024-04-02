import numpy as np
import pytest
import math
from sklearn.linear_model import LogisticRegression
from doubleml import DoubleMLLPQ, DoubleMLData
from doubleml.datasets import make_iivm_data
from doubleml.utils import DMLDummyClassifier
from ...tests._utils import draw_smpls


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_lpq_fixture(n_rep, normalize_ipw):
    ext_predictions = {"d": {}}
    np.random.seed(3141)
    data = make_iivm_data(theta=0.5, n_obs=2000, dim_x=10, alpha_x=1.0, return_type="DataFrame")

    dml_data = DoubleMLData(data, "y", "d", z_cols="z")
    all_smpls = draw_smpls(len(dml_data.y), 5, n_rep=n_rep, groups=dml_data.d)

    kwargs = {
        "obj_dml_data": dml_data,
        "score": "LPQ",
        "n_rep": n_rep,
        "normalize_ipw": normalize_ipw,
        "draw_sample_splitting": False,
    }

    ml_g = LogisticRegression()
    ml_m = LogisticRegression()

    dml_lpq = DoubleMLLPQ(ml_g=ml_g, ml_m=ml_m, **kwargs)
    dml_lpq.set_sample_splitting(all_smpls)

    np.random.seed(3141)
    dml_lpq.fit(store_predictions=True)

    ext_predictions["d"]["ml_m_z"] = dml_lpq.predictions["ml_m_z"][:, :, 0]
    ext_predictions["d"]["ml_m_d_z0"] = dml_lpq.predictions["ml_m_d_z0"][:, :, 0]
    ext_predictions["d"]["ml_m_d_z1"] = dml_lpq.predictions["ml_m_d_z1"][:, :, 0]
    ext_predictions["d"]["ml_g_du_z0"] = dml_lpq.predictions["ml_g_du_z0"][:, :, 0]
    ext_predictions["d"]["ml_g_du_z1"] = dml_lpq.predictions["ml_g_du_z1"][:, :, 0]

    dml_lpq_ext = DoubleMLLPQ(ml_g=DMLDummyClassifier(), ml_m=DMLDummyClassifier(), **kwargs)
    dml_lpq_ext.set_sample_splitting(all_smpls)

    np.random.seed(3141)
    dml_lpq_ext.fit(external_predictions=ext_predictions)

    res_dict = {"coef_normal": dml_lpq.coef[0], "coef_ext": dml_lpq_ext.coef[0]}

    return res_dict


@pytest.mark.ci
def test_doubleml_lpq_coef(doubleml_lpq_fixture):
    assert math.isclose(doubleml_lpq_fixture["coef_normal"], doubleml_lpq_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)
