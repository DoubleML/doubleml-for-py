import numpy as np
import pandas as pd
import pytest

from doubleml import DoubleMLClusterData, DoubleMLData
from doubleml.datasets import (
    _make_pliv_data,
    fetch_401K,
    fetch_bonus,
    make_confounded_irm_data,
    make_confounded_plr_data,
    make_heterogeneous_data,
    make_iivm_data,
    make_irm_data,
    make_irm_data_discrete_treatments,
    make_pliv_CHS2015,
    make_pliv_multiway_cluster_CKMS2021,
    make_plr_CCDDHNR2018,
    make_plr_turrell2018,
    make_ssm_data,
)

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
def test_make_irm_data_return_types():
    np.random.seed(3141)
    res = make_irm_data(n_obs=100, return_type="DoubleMLData")
    assert isinstance(res, DoubleMLData)
    res = make_irm_data(n_obs=100, return_type="DataFrame")
    assert isinstance(res, pd.DataFrame)
    x, y, d = make_irm_data(n_obs=100, return_type="array")
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_irm_data(n_obs=100, return_type="matrix")


@pytest.mark.ci
def test_make_iivm_data_return_types():
    np.random.seed(3141)
    res = make_iivm_data(n_obs=100, return_type="DoubleMLData")
    assert isinstance(res, DoubleMLData)
    res = make_iivm_data(n_obs=100, return_type="DataFrame")
    assert isinstance(res, pd.DataFrame)
    x, y, d, z = make_iivm_data(n_obs=100, return_type="array")
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(z, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_iivm_data(n_obs=100, return_type="matrix")


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
    res = make_pliv_multiway_cluster_CKMS2021(N=10, M=10, return_type="DoubleMLClusterData")
    assert isinstance(res, DoubleMLClusterData)
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


@pytest.fixture(scope="function", params=[True, False])
def linear(request):
    return request.param


@pytest.mark.ci
def test_make_confounded_irm_data_return_types(linear):
    np.random.seed(3141)
    res = make_confounded_irm_data(linear=linear)
    assert isinstance(res, dict)
    assert isinstance(res["x"], np.ndarray)
    assert isinstance(res["y"], np.ndarray)
    assert isinstance(res["d"], np.ndarray)

    assert isinstance(res["oracle_values"], dict)
    assert isinstance(res["oracle_values"]["g_long"], np.ndarray)
    assert isinstance(res["oracle_values"]["g_short"], np.ndarray)
    assert isinstance(res["oracle_values"]["m_long"], np.ndarray)
    assert isinstance(res["oracle_values"]["m_short"], np.ndarray)
    assert isinstance(res["oracle_values"]["gamma_a"], float)
    assert isinstance(res["oracle_values"]["beta_a"], float)
    assert isinstance(res["oracle_values"]["a"], np.ndarray)
    assert isinstance(res["oracle_values"]["y_0"], np.ndarray)
    assert isinstance(res["oracle_values"]["y_1"], np.ndarray)
    assert isinstance(res["oracle_values"]["z"], np.ndarray)
    assert isinstance(res["oracle_values"]["cf_y"], float)
    assert isinstance(res["oracle_values"]["cf_d_ate"], float)
    assert isinstance(res["oracle_values"]["cf_d_atte"], float)
    assert isinstance(res["oracle_values"]["rho_ate"], float)
    assert isinstance(res["oracle_values"]["rho_atte"], float)


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


@pytest.fixture(scope="function", params=[False, True])
def binary_treatment(request):
    return request.param


@pytest.fixture(scope="function", params=[1, 2])
def n_x(request):
    return request.param


@pytest.mark.ci
def test_make_heterogeneous_data_return_types(binary_treatment, n_x):
    np.random.seed(3141)
    res = make_heterogeneous_data(n_obs=100, n_x=n_x, binary_treatment=binary_treatment)
    assert isinstance(res, dict)
    assert isinstance(res["data"], pd.DataFrame)
    assert isinstance(res["effects"], np.ndarray)
    assert callable(res["treatment_effect"])

    # test input checks
    msg = "n_x must be either 1 or 2."
    with pytest.raises(AssertionError, match=msg):
        _ = make_heterogeneous_data(n_obs=100, n_x=0, binary_treatment=binary_treatment)
    msg = "support_size must be smaller than p."
    with pytest.raises(AssertionError, match=msg):
        _ = make_heterogeneous_data(n_obs=100, n_x=n_x, support_size=31, binary_treatment=binary_treatment)
    msg = "binary_treatment must be a boolean."
    with pytest.raises(AssertionError, match=msg):
        _ = make_heterogeneous_data(n_obs=100, n_x=n_x, binary_treatment=2)


@pytest.mark.ci
def test_make_ssm_data_return_types():
    np.random.seed(3141)
    res = make_ssm_data(n_obs=100)
    assert isinstance(res, DoubleMLData)
    res = make_ssm_data(n_obs=100, return_type="DataFrame")
    assert isinstance(res, pd.DataFrame)
    x, y, d, z, s = make_ssm_data(n_obs=100, return_type="array")
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(z, np.ndarray)
    assert isinstance(s, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_ssm_data(n_obs=100, return_type="matrix")


@pytest.fixture(scope="function", params=[3, 5])
def n_levels(request):
    return request.param


def test_make_data_discrete_treatments(n_levels):
    np.random.seed(3141)
    n = 100
    data_apo = make_irm_data_discrete_treatments(n_obs=n, n_levels=3)
    assert isinstance(data_apo, dict)
    assert isinstance(data_apo["y"], np.ndarray)
    assert isinstance(data_apo["d"], np.ndarray)
    assert isinstance(data_apo["x"], np.ndarray)
    assert isinstance(data_apo["oracle_values"], dict)

    assert isinstance(data_apo["oracle_values"]["cont_d"], np.ndarray)
    assert isinstance(data_apo["oracle_values"]["level_bounds"], np.ndarray)
    assert isinstance(data_apo["oracle_values"]["potential_level"], np.ndarray)
    assert isinstance(data_apo["oracle_values"]["ite"], np.ndarray)
    assert isinstance(data_apo["oracle_values"]["y0"], np.ndarray)

    msg = "n_levels must be at least 2."
    with pytest.raises(ValueError, match=msg):
        _ = make_irm_data_discrete_treatments(n_obs=n, n_levels=1)

    msg = "n_levels must be an integer."
    with pytest.raises(ValueError, match=msg):
        _ = make_irm_data_discrete_treatments(n_obs=n, n_levels=1.1)
