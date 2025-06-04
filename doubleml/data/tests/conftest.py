import numpy as np
import pandas as pd
import pytest

from doubleml.irm.datasets import make_irm_data
from doubleml.plm.datasets import make_plr_turrell2018


@pytest.fixture(scope="session", params=[(500, 10), (1000, 20), (1000, 100)])
def generate_data1(request):
    n_p = request.param
    np.random.seed(1111)
    # setting parameters
    n = n_p[0]
    p = n_p[1]
    theta = 0.5

    # generating data
    data = make_plr_turrell2018(n, p, theta, return_type=pd.DataFrame)

    return data


@pytest.fixture(scope="session", params=[(500, 10), (1000, 20)])
def generate_data_irm_w_missings(request):
    n_p = request.param
    np.random.seed(1111)
    # setting parameters
    n = n_p[0]
    p = n_p[1]
    theta = 0.5

    # generating data
    (x, y, d) = make_irm_data(n, p, theta, return_type="array")

    # randomly set some entries to np.nan
    ind = np.random.choice(np.arange(x.size), replace=False, size=int(x.size * 0.05))
    x[np.unravel_index(ind, x.shape)] = np.nan
    data = (x, y, d)

    return data
