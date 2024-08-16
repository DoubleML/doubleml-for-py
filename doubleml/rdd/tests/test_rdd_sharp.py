import pytest
import pandas as pd
import numpy as np

from doubleml import DoubleMLData
from doubleml.rdd.datasets import make_simple_rdd_data
from doubleml.rdd import RDFlex

from rdrobust import rdrobust
from sklearn.dummy import DummyRegressor, DummyClassifier

n = 500
data = make_simple_rdd_data(n_obs=n, fuzzy=False)
df = pd.DataFrame(
    np.column_stack((data['Y'], data['D'], data['score'], data['X'])),
    columns=['y', 'd', 'score'] + ['x' + str(i) for i in range(data['X'].shape[1])]
)

ml_g_dummy = DummyRegressor(strategy='constant', constant=0)
ml_m_dummy = DummyClassifier(strategy='constant', constant=0)


@pytest.fixture(scope='module',
                params=[-0.2, 0.0, 0.4])
def cutoff(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 2])
def p(request):
    return request.param


@pytest.fixture(scope='module')
def rdd_zero_predictions_fixture(cutoff, p):
    kwargs = {
        'p': p
    }

    # set the treatment indicator correctly based on the cutoff
    df['d'] = (data['score'] >= cutoff).astype(bool)
    dml_data = DoubleMLData(df, y_col='y', d_cols='d', s_col='score')

    dml_rdflex = RDFlex(
        dml_data,
        ml_g=ml_g_dummy,
        ml_m=ml_m_dummy,
        cutoff=cutoff,
        **kwargs)
    dml_rdflex.fit(n_iterations=1)

    rdrobust_model = rdrobust(y=df['y'], x=df['score'], c=cutoff, **kwargs)

    res_dict = {
        'dml_rdflex': dml_rdflex,
        'dml_coef': dml_rdflex.coef,
        'dml_se': dml_rdflex.se,
        'rdrobust_model': rdrobust_model,
        'rdrobust_coef': rdrobust_model.coef.values.flatten(),
        'rdrobust_se': rdrobust_model.se.values.flatten()
    }
    return res_dict


@pytest.mark.ci
def test_rdd_coef(rdd_zero_predictions_fixture):
    dml_coef = rdd_zero_predictions_fixture['dml_coef']
    rdrobust_coef = rdd_zero_predictions_fixture['rdrobust_coef']

    assert np.allclose(dml_coef, rdrobust_coef, rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_rdd_se(rdd_zero_predictions_fixture):
    dml_se = rdd_zero_predictions_fixture['dml_se']
    rdrobust_se = rdd_zero_predictions_fixture['rdrobust_se']

    assert np.allclose(dml_se, rdrobust_se, rtol=1e-9, atol=1e-4)
