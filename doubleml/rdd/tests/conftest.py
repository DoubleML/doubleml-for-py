import pytest

import numpy as np
import pandas as pd

from doubleml.rdd.datasets import make_simple_rdd_data


def test_dataframe(n_obs, fuzzy):
    data = make_simple_rdd_data(n_obs=n_obs, fuzzy=fuzzy)
    columns = ['y', 'd', 'score'] + ['x' + str(i) for i in range(data['X'].shape[1])]
    return pd.DataFrame(
        np.column_stack((data['Y'], data['D'], data['score'], data['X'])),
        columns=columns
    )


@pytest.fixture()
def rdd_sharp_data():
    return test_dataframe(n_obs=500, fuzzy=True)


@pytest.fixture()
def rdd_fuzzy_data():
    return test_dataframe(n_obs=500, fuzzy=True)


@pytest.fixture()
def rdd_fuzzy_left_data():
    return test_dataframe(n_obs=500, fuzzy=True)


@pytest.fixture()
def rdd_fuzzy_right_data():
    data = test_dataframe(n_obs=500, fuzzy=True)
    return data

