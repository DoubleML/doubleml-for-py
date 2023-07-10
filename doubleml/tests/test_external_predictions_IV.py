import numpy as np
import pytest
import math
from sklearn.linear_model import LinearRegression, LassoCV
from doubleml import DoubleMLPLIV, DoubleMLData
from doubleml.datasets import make_pliv_CHS2015

class dummy_learner:
    _estimator_type = "regressor"
    def fit(*args):
        raise AttributeError("Accessed fit method!")
    def predict(*args):
        raise AttributeError("Accessed predict method!")
    def set_params(*args):
        raise AttributeError("Accessed set_params method!")
    def get_params(*args, **kwargs):
        raise AttributeError("Accessed get_params method!")
    

@pytest.fixture(scope='module',
                params=['IV-type', 'partialling out'])
def score(request):
    return request.param

@pytest.fixture(scope='module',
                params=['dml1', 'dml2'])
def dml_procedure(request):
    return request.param

@pytest.fixture(scope='module',
                params=[1])
def n_rep(request):
    return request.param

@pytest.fixture(scope='module',
                params=[1, 3])
def dim_z(request):
    return request.param


@pytest.fixture(scope="module")
def adapted_doubleml_fixture(score, dml_procedure, n_rep):
    ext_predictions = {'d': {}}

    data = make_pliv_CHS2015(n_obs=500,
							 dim_x=20,
							 alpha=0.5,
							 dim_z=dim_z,
							 return_type="DataFrame")

    np.random.seed(3141)

    dml_data = DoubleMLData(data, 'y', 'd', z_cols=[f"Z{i}" for i in range(1, dim_z+1)])

    kwargs = {'obj_dml_data': dml_data,
              'score': score,
              'n_rep': n_rep,
              'dml_procedure': dml_procedure}
    
    if score == 'IV-type':
        kwargs['ml_g'] = LinearRegression()
    
    DMLPLIV = DoubleMLPLIV(ml_m=LinearRegression(),
                           ml_l=LinearRegression(),
                           ml_r=LinearRegression(),
                           **kwargs)
    np.random.seed(3141)

    DMLPLIV.fit(store_predictions=True)

    ext_predictions['d']['ml_l'] = DMLPLIV.predictions['ml_l'].squeeze()
    ext_predictions['d']['ml_r'] = DMLPLIV.predictions['ml_r'].squeeze()

    if dimz == 1:
        ext_predictions['d']['ml_m'] = DMLPLIV.predictions['ml_m'].squeeze()
    else:
        for instr in range(dimz):
            ext_predictions['d']['ml_m_' + 'Z' + str(instr+1)] = DMLPLIV.predictions['ml_m_' + 'Z' + str(instr+1)].squeeze()

        if score == 'IV-type':
            kwargs['ml_g'] = dummy_learner()
            ext_predictions['d']['ml_g'] = DMLPLIV.predictions['ml_g'].squeeze()


    DMLPLIV_ext = DoubleMLPLIV(ml_m=dummy_learner(),
                               ml_l=dummy_learner(),
                               ml_r=dummy_learner(),
                               **kwargs)

    np.random.seed(3141)
    DMLPLR_ext.fit(external_predictions=ext_predictions)

    res_dict = {'coef_normal': DMLPLIV.coef,
                'coef_ext': DMLPLIV_ext.coef}

    return res_dict

@pytest.mark.ci
def test_adapted_doubleml_coef(adapted_doubleml_fixture):
    assert math.isclose(adapted_doubleml_fixture['coef_normal'],
                        adapted_doubleml_fixture['coef_ext'],
                        rel_tol=1e-9, abs_tol=1e-4)