import numpy as np
import pandas as pd
import pytest

from doubleml import DoubleMLData
from doubleml.plm.datasets import (
    _make_pliv_data,
    make_confounded_plr_data,
    make_lplr_LZZ2020,
    make_pliv_CHS2015,
    make_pliv_multiway_cluster_CKMS2021,
    make_plr_CCDDHNR2018,
    make_plr_turrell2018,
)

msg_inv_return_type = "Invalid return_type."


@pytest.mark.ci
def test_make_plr_CCDDHNR2018_return_types():
    np.random.seed(3141)
    res = make_plr_CCDDHNR2018(n_obs=100, return_type=DoubleMLData)
    assert isinstance(res, DoubleMLData)
    res = make_plr_CCDDHNR2018(n_obs=100, return_type=pd.DataFrame)
    assert isinstance(res, pd.DataFrame)
    x, y, d = make_plr_CCDDHNR2018(n_obs=100, return_type=np.ndarray)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_plr_CCDDHNR2018(n_obs=100, return_type="matrix")


@pytest.mark.ci
def test_make_plr_turrell2018_return_types():
    np.random.seed(3141)
    res = make_plr_turrell2018(n_obs=100, return_type="DoubleMLData")
    assert isinstance(res, DoubleMLData)
    res = make_plr_turrell2018(n_obs=100, return_type="DataFrame")
    assert isinstance(res, pd.DataFrame)
    x, y, d = make_plr_turrell2018(n_obs=100, return_type="array")
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_plr_turrell2018(n_obs=100, return_type="matrix")


@pytest.mark.ci
def test_make_confounded_plr_data_return_types():
    np.random.seed(3141)
    res = make_confounded_plr_data(theta=5.0)
    assert isinstance(res, dict)
    assert isinstance(res["x"], np.ndarray)
    assert isinstance(res["y"], np.ndarray)
    assert isinstance(res["d"], np.ndarray)

    assert isinstance(res["oracle_values"], dict)
    assert isinstance(res["oracle_values"]["g_long"], np.ndarray)
    assert isinstance(res["oracle_values"]["g_short"], np.ndarray)
    assert isinstance(res["oracle_values"]["m_long"], np.ndarray)
    assert isinstance(res["oracle_values"]["m_short"], np.ndarray)
    assert isinstance(res["oracle_values"]["theta"], float)
    assert isinstance(res["oracle_values"]["gamma_a"], float)
    assert isinstance(res["oracle_values"]["beta_a"], float)
    assert isinstance(res["oracle_values"]["a"], np.ndarray)
    assert isinstance(res["oracle_values"]["z"], np.ndarray)


@pytest.mark.ci
def test_make_pliv_data_return_types():
    np.random.seed(3141)
    res = _make_pliv_data(n_obs=100, return_type="DoubleMLData")
    assert isinstance(res, DoubleMLData)
    res = _make_pliv_data(n_obs=100, return_type="DataFrame")
    assert isinstance(res, pd.DataFrame)
    x, y, d, z = _make_pliv_data(n_obs=100, return_type="array")
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(z, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = _make_pliv_data(n_obs=100, return_type="matrix")


@pytest.mark.ci
def test_make_pliv_CHS2015_return_types():
    np.random.seed(3141)
    res = make_pliv_CHS2015(n_obs=100, return_type="DoubleMLData")
    assert isinstance(res, DoubleMLData)
    res = make_pliv_CHS2015(n_obs=100, return_type="DataFrame")
    assert isinstance(res, pd.DataFrame)
    x, y, d, z = make_pliv_CHS2015(n_obs=100, return_type="array")
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(z, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_pliv_CHS2015(n_obs=100, return_type="matrix")


@pytest.mark.ci
def test_make_pliv_multiway_cluster_CKMS2021_return_types():
    np.random.seed(3141)
    res = make_pliv_multiway_cluster_CKMS2021(N=10, M=10, return_type="DoubleMLData")
    assert isinstance(res, DoubleMLData)
    res = make_pliv_multiway_cluster_CKMS2021(N=10, M=10, return_type="DataFrame")
    assert isinstance(res, pd.DataFrame)
    x, y, d, cluster_vars, z = make_pliv_multiway_cluster_CKMS2021(N=10, M=10, return_type="array")
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(cluster_vars, np.ndarray)
    assert isinstance(z, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_pliv_multiway_cluster_CKMS2021(N=10, M=10, return_type="matrix")


@pytest.mark.ci
def test_make_lplr_LZZ2020_return_types():
    np.random.seed(3141)
    res = make_lplr_LZZ2020(n_obs=100, return_type="DoubleMLData")
    assert isinstance(res, DoubleMLData)
    res = make_lplr_LZZ2020(n_obs=100, return_type="DataFrame")
    assert isinstance(res, pd.DataFrame)
    x, y, d, z = make_lplr_LZZ2020(n_obs=100, return_type="array")
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(z, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_lplr_LZZ2020(n_obs=100, return_type="matrix")


@pytest.mark.ci
def test_make_lplr_LZZ2020_variants():
    np.random.seed(3141)
    res = make_lplr_LZZ2020(n_obs=100, treatment="binary")
    assert np.array_equal(np.unique(res.d), np.array([0, 1]))
    res = make_lplr_LZZ2020(n_obs=100, treatment="binary_unbalanced")
    assert np.array_equal(np.unique(res.d), np.array([0, 1]))
    res = make_lplr_LZZ2020(n_obs=100, treatment="continuous")
    assert len(np.unique(res.d)) == 100

    msg = "Invalid treatment type."
    with pytest.raises(ValueError, match=msg):
        _ = make_lplr_LZZ2020(n_obs=100, treatment="colors")

    res = make_lplr_LZZ2020(n_obs=100, balanced_r0=False)
    _, y_unique = np.unique(res.y, return_counts=True)
    assert np.abs(y_unique[0] - y_unique[1]) > 10
