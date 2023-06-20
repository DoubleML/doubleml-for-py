import numpy as np
import pytest
import math
from sklearn.linear_model import LinearRegression, LassoCV
from doubleml import DoubleMLPLR, DoubleMLData
from doubleml.datasets import make_plr_CCDDHNR2018

# @pytest.fixture(scope='module',
#                 params=[LinearRegression(),
#                         LassoCV()])
# def learner(request):
#     return request.param

@pytest.fixture(scope='module',
                params=['IV-type', 'partialling out'])
def score(request):
    return request.param

@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param

@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module")
def adapted_doubleml_fixture(score, dml_procedure, n_rep):
    ext_predictions = {'d': {}}

    x, y, d = make_plr_CCDDHNR2018(n_obs=500,
                                   dim_x=20,
                                   alpha=0.5,
                                   return_type="np.array")

    # lm_m1 = LinearRegression()
    # lm_l1 = LinearRegression()

    np.random.seed(3141)

    # lm_m1.fit(x, d)
    # ext_predictions['d']['ml_m'] = np.stack([lm_m1.predict(x) for _ in range(n_rep)], axis=1)

    # lm_l1.fit(x, y)
    # ext_predictions['d']['ml_l'] = np.stack([lm_l1.predict(x) for _ in range(n_rep)], axis=1)

    dml_data = DoubleMLData.from_arrays(x=x, y=y, d=d)

    DMLPLR = DoubleMLPLR(obj_dml_data=dml_data,
                         ml_m=LinearRegression(),
                         ml_l=LinearRegression(),
                         score=score,
                         n_rep=n_rep,
                         dml_procedure=dml_procedure)
    np.random.seed(3141)

    DMLPLR.fit(store_predictions=True)

    ext_predictions['d']['ml_m'] = DMLPLR.predictions['ml_m'].squeeze()
    ext_predictions['d']['ml_l'] = DMLPLR.predictions['ml_l'].squeeze()


    DMLPLR_ext = DoubleMLPLR(obj_dml_data=dml_data,
                             ml_m=LinearRegression(),
                             ml_l=LinearRegression(),
                             score=score,
                             n_rep=n_rep,
                             dml_procedure=dml_procedure)

    np.random.seed(3141)
    DMLPLR_ext.fit(external_predictions=ext_predictions)

    res_dict = {'coef_normal': DMLPLR.coef,
                'coef_ext': DMLPLR_ext.coef}

    return res_dict

@pytest.mark.ci
def test_adapted_doubleml_coef(adapted_doubleml_fixture):
    assert math.isclose(adapted_doubleml_fixture['coef_normal'],
                        adapted_doubleml_fixture['coef_ext'],
                        rel_tol=1e-9, abs_tol=1e-4)