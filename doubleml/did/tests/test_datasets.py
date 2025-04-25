import numpy as np
import pandas as pd
import pytest

from doubleml import DoubleMLData
from doubleml.did.datasets import make_did_CS2021, make_did_SZ2020

msg_inv_return_type = "Invalid return_type."


@pytest.fixture(scope="function", params=[False, True])
def cross_sectional(request):
    return request.param


@pytest.fixture(scope="function", params=[1, 2, 3, 4, 5, 6])
def dgp_type(request):
    return request.param


@pytest.mark.ci
def test_make_did_SZ2020_return_types(cross_sectional, dgp_type):
    np.random.seed(3141)
    res = make_did_SZ2020(n_obs=100, dgp_type=dgp_type, cross_sectional_data=cross_sectional, return_type=DoubleMLData)
    assert isinstance(res, DoubleMLData)
    res = make_did_SZ2020(n_obs=100, dgp_type=dgp_type, cross_sectional_data=cross_sectional, return_type=pd.DataFrame)
    assert isinstance(res, pd.DataFrame)
    if cross_sectional:
        x, y, d, t = make_did_SZ2020(
            n_obs=100, dgp_type=dgp_type, cross_sectional_data=cross_sectional, return_type=np.ndarray
        )
        assert isinstance(t, np.ndarray)
    else:
        x, y, d, _ = make_did_SZ2020(
            n_obs=100, dgp_type=dgp_type, cross_sectional_data=cross_sectional, return_type=np.ndarray
        )
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_did_SZ2020(n_obs=100, dgp_type=dgp_type, cross_sectional_data=cross_sectional, return_type="matrix")
    msg = "The dgp_type is not valid."
    with pytest.raises(ValueError, match=msg):
        _ = make_did_SZ2020(n_obs=100, dgp_type="5", cross_sectional_data=cross_sectional, return_type="matrix")


@pytest.fixture(scope="function", params=[True, False])
def include_never_treated(request):
    return request.param


@pytest.fixture(scope="function", params=["datetime", "float"])
def time_type(request):
    return request.param


@pytest.fixture(scope="function", params=[0, 2])
def anticipation_periods(request):
    return request.param


@pytest.mark.ci
def test_make_did_CS2021_return_types(dgp_type, include_never_treated, time_type, anticipation_periods):
    np.random.seed(3141)
    df = make_did_CS2021(
        n_obs=100,
        dgp_type=dgp_type,
        include_never_treated=include_never_treated,
        time_type=time_type,
        anticipation_periods=anticipation_periods,
    )
    assert isinstance(df, pd.DataFrame)


@pytest.mark.ci
def test_make_did_CS2021_exceptions():
    msg = r"time_type must be one of \('datetime', 'float'\). Got 2."
    with pytest.raises(ValueError, match=msg):
        _ = make_did_CS2021(n_obs=100, time_type=2)
