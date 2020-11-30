import pytest
import numpy as np

from doubleml.double_ml_data import DoubleMLData

from doubleml.tests.helper_general import get_n_datasets


# number of datasets per dgp
n_datasets = get_n_datasets()

@pytest.fixture(scope='module',
                params = range(n_datasets))
def idx(request):
    return request.param


@pytest.fixture(scope="module")
def dml_data_fixture(generate_data1, idx):
    data = generate_data1[idx]
    np.random.seed(3141)
    x_cols = data.columns[data.columns.str.startswith('X')].tolist()
    
    obj_from_np = DoubleMLData.from_arrays(data.loc[:, x_cols].values,
                                           data['y'].values, data['d'].values)

    obj_from_pd = DoubleMLData(data, 'y', ['d'], x_cols)
    
    return {'obj_from_np': obj_from_np,
            'obj_from_pd': obj_from_pd}


@pytest.mark.ci
def test_dml_data_x(dml_data_fixture):
    assert np.allclose(dml_data_fixture['obj_from_np'].x,
                       dml_data_fixture['obj_from_pd'].x,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_data_y(dml_data_fixture):
    assert np.allclose(dml_data_fixture['obj_from_np'].y,
                       dml_data_fixture['obj_from_pd'].y,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_data_d(dml_data_fixture):
    assert np.allclose(dml_data_fixture['obj_from_np'].d,
                       dml_data_fixture['obj_from_pd'].d,
                       rtol=1e-9, atol=1e-4)
