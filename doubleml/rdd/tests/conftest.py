import pytest

import numpy as np
import pandas as pd

from doubleml.rdd.datasets import make_simple_rdd_data
from doubleml import DoubleMLData
from doubleml.rdd import RDFlex

from rdrobust import rdrobust

from sklearn.dummy import DummyRegressor, DummyClassifier


DATA_SIZE = 500


ml_g_dummy = DummyRegressor(strategy='constant', constant=0)
ml_m_dummy = DummyClassifier(strategy='constant', constant=0)


@pytest.fixture(scope='module')
def predict_dummy():
    """
    - make predictions using rd-flex with constant model
    - make predictions using rdrobust as a reference
    """
    def _predict_dummy(data: DoubleMLData, cutoff, alpha, n_rep, p):
        dml_rdflex = RDFlex(
            data,
            ml_g=ml_g_dummy,
            ml_m=ml_m_dummy,
            cutoff=cutoff,
            n_rep=n_rep,
            p=p
        )
        dml_rdflex.fit(n_iterations=1)
        ci_manual = dml_rdflex.confint(level=1-alpha)

        rdrobust_model = rdrobust(
            y=data.y,
            x=data.s,
            c=cutoff,
            level=100*(1-alpha),
            p=p
        )

        reference = {
            'model': rdrobust_model,
            'coef': rdrobust_model.coef.values.flatten(),
            'se': rdrobust_model.se.values.flatten(),
            'ci': rdrobust_model.ci.values
        }

        actual = {
            'model': dml_rdflex,
            'coef': dml_rdflex.coef,
            'se': dml_rdflex.se,
            'ci': ci_manual,
        }
        return reference, actual

    return _predict_dummy


def defier_mask(fuzzy, data, actual_cutoff):
    if fuzzy == 'left':
        # right defiers (not treated even if score suggested it
        return (data['D'] == 0) & (data['score'] >= actual_cutoff)
    elif fuzzy == 'right':
        # left defiers (treated even if score not suggested it
        return (data['D'] == 1) & (data['score'] < actual_cutoff)
    elif fuzzy in ['both', 'none']:
        return None
    raise ValueError(f'Invalid type of fuzzyness {fuzzy}')


def generate_data(
    n_obs: int,
    fuzzy: str,
    cutoff: float,
):
    data = make_simple_rdd_data(
        n_obs=n_obs,
        fuzzy=fuzzy in ['both', 'left', 'right'],
        cutoff=cutoff
    )

    mask = defier_mask(fuzzy, data, cutoff)
    if mask is not None:
        data = {k: v[~mask] for k, v in data.items() if k != 'oracle_values'}

    columns = ['y', 'd', 'score'] + ['x' + str(i) for i in range(data['X'].shape[1])]
    df = pd.DataFrame(
        np.column_stack((data['Y'], data['D'], data['score'], data['X'])),
        columns=columns
    )
    return DoubleMLData(df, y_col='y', d_cols='d', s_col='score')


@pytest.fixture(scope='module')
def rdd_sharp_data():
    def _rdd_sharp_data(cutoff):
        return generate_data(n_obs=DATA_SIZE, fuzzy='none', cutoff=cutoff)
    return _rdd_sharp_data


@pytest.fixture(scope='module')
def rdd_fuzzy_data():
    def _rdd_fuzzy_data(cutoff):
        return generate_data(n_obs=DATA_SIZE, fuzzy='both', cutoff=cutoff)
    return _rdd_fuzzy_data


@pytest.fixture(scope='module')
def rdd_fuzzy_left_data():
    def _rdd_fuzzy_left_data(cutoff):
        return generate_data(n_obs=DATA_SIZE, fuzzy='left', cutoff=cutoff)
    return _rdd_fuzzy_left_data


@pytest.fixture(scope='module')
def rdd_fuzzy_right_data():
    def _rdd_fuzzy_right_data(cutoff):
        data = generate_data(n_obs=DATA_SIZE, fuzzy='left', cutoff=cutoff)
        return data
    return _rdd_fuzzy_right_data
