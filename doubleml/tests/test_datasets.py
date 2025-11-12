import pandas as pd
import pytest

from doubleml import DoubleMLData
from doubleml.datasets import fetch_401K, fetch_bonus

msg_inv_return_type = "Invalid return_type."


def test_fetch_401K_return_types():
    res = fetch_401K("DoubleMLData")
    assert isinstance(res, DoubleMLData)
    res = fetch_401K("DataFrame")
    assert isinstance(res, pd.DataFrame)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = fetch_401K("matrix")


def test_fetch_401K_poly():
    msg = "polynomial_features os not implemented yet for fetch_401K."
    with pytest.raises(NotImplementedError, match=msg):
        _ = fetch_401K(polynomial_features=True)


def test_fetch_bonus_return_types():
    res = fetch_bonus("DoubleMLData")
    assert isinstance(res, DoubleMLData)
    res = fetch_bonus("DataFrame")
    assert isinstance(res, pd.DataFrame)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = fetch_bonus("matrix")


def test_fetch_bonus_poly():
    data_bonus_wo_poly = fetch_bonus(polynomial_features=False)
    n_x = len(data_bonus_wo_poly.x_cols)
    data_bonus_w_poly = fetch_bonus(polynomial_features=True)
    assert len(data_bonus_w_poly.x_cols) == ((n_x + 1) * n_x / 2 + n_x)
