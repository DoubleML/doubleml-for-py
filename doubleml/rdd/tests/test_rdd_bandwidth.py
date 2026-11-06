"""RDFlex reports the bandwidth that was used in the final estimation."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

import doubleml as dml
from doubleml.rdd import RDFlex
from doubleml.rdd.datasets import make_simple_rdd_data


@pytest.fixture(scope="module")
def dml_data():
    np.random.seed(3141)
    data = make_simple_rdd_data(n_obs=1000, fuzzy=False)
    return dml.DoubleMLRDDData.from_arrays(data["X"], data["Y"], data["D"], score=data["score"])


@pytest.mark.ci_rdd
@pytest.mark.parametrize("n_iterations", [1, 2])
def test_rdd_bandwidth(dml_data, n_iterations):
    """`h` holds the bandwidth of the final estimation for each repetition, also without iterations."""
    np.random.seed(42)
    dml_rdflex = RDFlex(dml_data, ml_g=LinearRegression(), n_rep=2).fit(n_iterations=n_iterations)

    h_final = np.array([rdd_res.bws.loc["h"].max() for rdd_res in dml_rdflex._rdd_obj])
    assert np.all(np.isfinite(dml_rdflex.h))
    np.testing.assert_allclose(dml_rdflex.h, h_final)


@pytest.mark.ci_rdd
@pytest.mark.parametrize("n_iterations", [1, 2])
def test_rdd_bandwidth_user_given(dml_data, n_iterations):
    """A bandwidth passed to rdrobust via ``h`` is reported as `h`."""
    np.random.seed(42)
    with pytest.warns(UserWarning, match="Key-worded arguments contain"):
        dml_rdflex = RDFlex(dml_data, ml_g=LinearRegression(), n_rep=2, h=0.5)
    dml_rdflex.fit(n_iterations=n_iterations)

    np.testing.assert_allclose(dml_rdflex.h, [0.5, 0.5])
