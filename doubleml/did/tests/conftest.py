import numpy as np
import pytest

from doubleml.datasets import make_did_SZ2020


@pytest.fixture(scope='session',
                params=[(500, 1),
                        (1000, 1),
                        (1000, 2)])
def generate_data_did(request):
    params = request.param
    np.random.seed(1111)
    # setting parameters
    n = params[0]
    dpg = params[1]

    # generating data
    data = make_did_SZ2020(n, dgp_type=dpg, return_type='array')

    return data


@pytest.fixture(scope='session',
                params=[(500, 1),
                        (1000, 1),
                        (1000, 2)])
def generate_data_did_cs(request):
    params = request.param
    np.random.seed(1111)
    # setting parameters
    n = params[0]
    dpg = params[1]

    # generating data
    data = make_did_SZ2020(n, dgp_type=dpg, cross_sectional_data=True, return_type='array')

    return data
