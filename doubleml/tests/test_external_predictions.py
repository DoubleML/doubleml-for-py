import numpy as np
import pytest
import math
from sklearn.linear_model import LinearRegression, LassoCV
from doubleml import DoubleMLPLR, DoubleMLData
from doubleml.datasets import make_plr_CCDDHNR2018
from doubleml.utils import dummy_regressor
    

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

    np.random.seed(3141)

    dml_data = DoubleMLData.from_arrays(x=x, y=y, d=d)

    kwargs = {'obj_dml_data': dml_data,
              'score': score,
              'n_rep': n_rep,
              'dml_procedure': dml_procedure}

    if score == 'IV-type':
        kwargs['ml_g'] = LinearRegression()

    DMLPLR = DoubleMLPLR(ml_m=LinearRegression(),
                         ml_l=LinearRegression(),
                         **kwargs)
    np.random.seed(3141)

    DMLPLR.fit(store_predictions=True)

    ext_predictions['d']['ml_m'] = DMLPLR.predictions['ml_m'][:, :, 0]
    ext_predictions['d']['ml_l'] = DMLPLR.predictions['ml_l'][:, :, 0]

    if score == 'IV-type':
        kwargs['ml_g'] = dummy_regressor()
        ext_predictions['d']['ml_g'] = DMLPLR.predictions['ml_g'][:, :, 0]


    DMLPLR_ext = DoubleMLPLR(ml_m=dummy_regressor(),
                             ml_l=dummy_regressor(),
                             **kwargs)

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