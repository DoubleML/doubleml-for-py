import numpy as np
import pytest
import math
from sklearn.linear_model import LogisticRegression
from doubleml import DoubleMLPQ, DoubleMLData
from doubleml.datasets import make_irm_data
from doubleml.utils import dummy_regressor, dummy_classifier


@pytest.fixture(scope="module", params=["dml1", "dml2"])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope="module")
def doubleml_pq_fixture(dml_procedure, n_rep, normalize_ipw):
    ext_predictions = {"d": {}}
    np.random.seed(3141)
    data = make_irm_data(theta=0.5, n_obs=500, dim_x=20, return_type="DataFrame")

    dml_data = DoubleMLData(data, "y", "d")

    kwargs = {
        "obj_dml_data": dml_data,
        "score": "PQ",
        "n_rep": n_rep,
        "dml_procedure": dml_procedure,
        "normalize_ipw": normalize_ipw,
    }

    ml_g = LogisticRegression()
    ml_m = LogisticRegression()

    DMLPQ = DoubleMLPQ(ml_g=ml_g, ml_m=ml_m, **kwargs)
    np.random.seed(3141)

    DMLPQ.fit(store_predictions=True)

    ext_predictions["d"]["ml_g"] = DMLPQ.predictions["ml_g"][:, :, 0]
    ext_predictions["d"]["ml_m"] = DMLPQ.predictions["ml_m"][:, :, 0]

    DMLPLQ_ext = DoubleMLPQ(ml_g=dummy_classifier(), ml_m=dummy_classifier(), **kwargs)

    np.random.seed(3141)
    DMLPLQ_ext.fit(external_predictions=ext_predictions)

    res_dict = {"coef_normal": DMLPQ.coef, "coef_ext": DMLPLQ_ext.coef}

    return res_dict


@pytest.mark.ci
def test_doubleml_pq_coef(doubleml_pq_fixture):
    assert math.isclose(doubleml_pq_fixture["coef_normal"], doubleml_pq_fixture["coef_ext"], rel_tol=1e-9, abs_tol=1e-4)
