import pytest
import pandas as pd
import numpy as np

from doubleml import DoubleMLData, DoubleMLClusterData
from doubleml.datasets import fetch_401K, fetch_bonus, make_plr_CCDDHNR2018, make_plr_turrell2018, \
    make_irm_data, make_iivm_data, _make_pliv_data, make_pliv_CHS2015, make_pliv_multiway_cluster_CKMS2021, \
    make_did_SZ2020, make_confounded_irm_data, make_confounded_plr_data, make_heterogeneous_data, make_ssm_data

msg_inv_return_type = 'Invalid return_type.'


def test_fetch_401K_return_types():
    res = fetch_401K('DoubleMLData')
    assert isinstance(res, DoubleMLData)
    res = fetch_401K('DataFrame')
    assert isinstance(res, pd.DataFrame)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = fetch_401K('matrix')


def test_fetch_401K_poly():
    msg = 'polynomial_features os not implemented yet for fetch_401K.'
    with pytest.raises(NotImplementedError, match=msg):
        _ = fetch_401K(polynomial_features=True)


def test_fetch_bonus_return_types():
    res = fetch_bonus('DoubleMLData')
    assert isinstance(res, DoubleMLData)
    res = fetch_bonus('DataFrame')
    assert isinstance(res, pd.DataFrame)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = fetch_bonus('matrix')


def test_fetch_bonus_poly():
    data_bonus_wo_poly = fetch_bonus(polynomial_features=False)
    n_x = len(data_bonus_wo_poly.x_cols)
    data_bonus_w_poly = fetch_bonus(polynomial_features=True)
    assert len(data_bonus_w_poly.x_cols) == ((n_x+1) * n_x / 2 + n_x)


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
        _ = make_plr_CCDDHNR2018(n_obs=100, return_type='matrix')


@pytest.mark.ci
def test_make_plr_turrell2018_return_types():
    np.random.seed(3141)
    res = make_plr_turrell2018(n_obs=100, return_type='DoubleMLData')
    assert isinstance(res, DoubleMLData)
    res = make_plr_turrell2018(n_obs=100, return_type='DataFrame')
    assert isinstance(res, pd.DataFrame)
    x, y, d = make_plr_turrell2018(n_obs=100, return_type='array')
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_plr_turrell2018(n_obs=100, return_type='matrix')


@pytest.mark.ci
def test_make_irm_data_return_types():
    np.random.seed(3141)
    res = make_irm_data(n_obs=100, return_type='DoubleMLData')
    assert isinstance(res, DoubleMLData)
    res = make_irm_data(n_obs=100, return_type='DataFrame')
    assert isinstance(res, pd.DataFrame)
    x, y, d = make_irm_data(n_obs=100, return_type='array')
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_irm_data(n_obs=100, return_type='matrix')


@pytest.mark.ci
def test_make_iivm_data_return_types():
    np.random.seed(3141)
    res = make_iivm_data(n_obs=100, return_type='DoubleMLData')
    assert isinstance(res, DoubleMLData)
    res = make_iivm_data(n_obs=100, return_type='DataFrame')
    assert isinstance(res, pd.DataFrame)
    x, y, d, z = make_iivm_data(n_obs=100, return_type='array')
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(z, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_iivm_data(n_obs=100, return_type='matrix')


@pytest.mark.ci
def test_make_pliv_data_return_types():
    np.random.seed(3141)
    res = _make_pliv_data(n_obs=100, return_type='DoubleMLData')
    assert isinstance(res, DoubleMLData)
    res = _make_pliv_data(n_obs=100, return_type='DataFrame')
    assert isinstance(res, pd.DataFrame)
    x, y, d, z = _make_pliv_data(n_obs=100, return_type='array')
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(z, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = _make_pliv_data(n_obs=100, return_type='matrix')


@pytest.mark.ci
def test_make_pliv_CHS2015_return_types():
    np.random.seed(3141)
    res = make_pliv_CHS2015(n_obs=100, return_type='DoubleMLData')
    assert isinstance(res, DoubleMLData)
    res = make_pliv_CHS2015(n_obs=100, return_type='DataFrame')
    assert isinstance(res, pd.DataFrame)
    x, y, d, z = make_pliv_CHS2015(n_obs=100, return_type='array')
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(z, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_pliv_CHS2015(n_obs=100, return_type='matrix')


@pytest.mark.ci
def test_make_pliv_multiway_cluster_CKMS2021_return_types():
    np.random.seed(3141)
    res = make_pliv_multiway_cluster_CKMS2021(N=10, M=10, return_type='DoubleMLClusterData')
    assert isinstance(res, DoubleMLClusterData)
    res = make_pliv_multiway_cluster_CKMS2021(N=10, M=10, return_type='DataFrame')
    assert isinstance(res, pd.DataFrame)
    x, y, d, cluster_vars, z = make_pliv_multiway_cluster_CKMS2021(N=10, M=10, return_type='array')
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(cluster_vars, np.ndarray)
    assert isinstance(z, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_pliv_multiway_cluster_CKMS2021(N=10, M=10, return_type='matrix')


@pytest.fixture(scope='function',
                params=[False, True])
def cross_sectional(request):
    return request.param


@pytest.fixture(scope='function',
                params=[1, 2, 3, 4, 5, 6])
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
        x, y, d, t = make_did_SZ2020(n_obs=100, dgp_type=dgp_type, cross_sectional_data=cross_sectional,
                                     return_type=np.ndarray)
        assert isinstance(t, np.ndarray)
    else:
        x, y, d = make_did_SZ2020(n_obs=100, dgp_type=dgp_type, cross_sectional_data=cross_sectional, return_type=np.ndarray)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_did_SZ2020(n_obs=100, dgp_type=dgp_type, cross_sectional_data=cross_sectional, return_type='matrix')
    msg = 'The dgp_type is not valid.'
    with pytest.raises(ValueError, match=msg):
        _ = make_did_SZ2020(n_obs=100, dgp_type="5", cross_sectional_data=cross_sectional, return_type='matrix')


@pytest.mark.ci
def test_make_confounded_irm_data_return_types():
    np.random.seed(3141)
    res = make_confounded_irm_data()
    assert isinstance(res, dict)
    assert isinstance(res['x'], np.ndarray)
    assert isinstance(res['y'], np.ndarray)
    assert isinstance(res['d'], np.ndarray)

    assert isinstance(res['oracle_values'], dict)
    assert isinstance(res['oracle_values']['g_long'], np.ndarray)
    assert isinstance(res['oracle_values']['g_short'], np.ndarray)
    assert isinstance(res['oracle_values']['m_long'], np.ndarray)
    assert isinstance(res['oracle_values']['m_short'], np.ndarray)
    assert isinstance(res['oracle_values']['gamma_a'], float)
    assert isinstance(res['oracle_values']['beta_a'], float)
    assert isinstance(res['oracle_values']['a'], np.ndarray)
    assert isinstance(res['oracle_values']['y0'], np.ndarray)
    assert isinstance(res['oracle_values']['y1'], np.ndarray)
    assert isinstance(res['oracle_values']['z'], np.ndarray)


@pytest.mark.ci
def test_make_confounded_plr_data_return_types():
    np.random.seed(3141)
    res = make_confounded_plr_data(theta=5.0)
    assert isinstance(res, dict)
    assert isinstance(res['x'], np.ndarray)
    assert isinstance(res['y'], np.ndarray)
    assert isinstance(res['d'], np.ndarray)

    assert isinstance(res['oracle_values'], dict)
    assert isinstance(res['oracle_values']['g_long'], np.ndarray)
    assert isinstance(res['oracle_values']['g_short'], np.ndarray)
    assert isinstance(res['oracle_values']['m_long'], np.ndarray)
    assert isinstance(res['oracle_values']['m_short'], np.ndarray)
    assert isinstance(res['oracle_values']['theta'], float)
    assert isinstance(res['oracle_values']['gamma_a'], float)
    assert isinstance(res['oracle_values']['beta_a'], float)
    assert isinstance(res['oracle_values']['a'], np.ndarray)
    assert isinstance(res['oracle_values']['z'], np.ndarray)


@pytest.fixture(scope='function',
                params=[False, True])
def binary_treatment(request):
    return request.param


@pytest.fixture(scope='function',
                params=[1, 2])
def n_x(request):
    return request.param


@pytest.mark.ci
def test_make_heterogeneous_data_return_types(binary_treatment, n_x):
    np.random.seed(3141)
    res = make_heterogeneous_data(n_obs=100, n_x=n_x, binary_treatment=binary_treatment)
    assert isinstance(res, dict)
    assert isinstance(res['data'], pd.DataFrame)
    assert isinstance(res['effects'], np.ndarray)
    assert callable(res['treatment_effect'])

    # test input checks
    msg = 'n_x must be either 1 or 2.'
    with pytest.raises(AssertionError, match=msg):
        _ = make_heterogeneous_data(n_obs=100, n_x=0, binary_treatment=binary_treatment)
    msg = 'support_size must be smaller than p.'
    with pytest.raises(AssertionError, match=msg):
        _ = make_heterogeneous_data(n_obs=100, n_x=n_x, support_size=31, binary_treatment=binary_treatment)
    msg = 'binary_treatment must be a boolean.'
    with pytest.raises(AssertionError, match=msg):
        _ = make_heterogeneous_data(n_obs=100, n_x=n_x, binary_treatment=2)


@pytest.mark.ci
def test_make_ssm_data_return_types():
    np.random.seed(3141)
    res = make_ssm_data(n_obs=100)
    assert isinstance(res, DoubleMLData)
    res = make_ssm_data(n_obs=100, return_type='DataFrame')
    assert isinstance(res, pd.DataFrame)
    x, y, d, z, s = make_ssm_data(n_obs=100, return_type='array')
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(z, np.ndarray)
    assert isinstance(s, np.ndarray)
    with pytest.raises(ValueError, match=msg_inv_return_type):
        _ = make_ssm_data(n_obs=100, return_type='matrix')
