import numpy as np
import pytest
import math
import scipy

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

from dml.double_ml_pliv import DoubleMLPLIV

from dml.tests.helper_general import get_n_datasets
from dml.tests.helper_pliv_manual import pliv_dml1, pliv_dml2, fit_nuisance_pliv, boot_pliv


# number of datasets per dgp
n_datasets = get_n_datasets()


@pytest.mark.parametrize('idx', range(n_datasets))
@pytest.mark.parametrize('learner', [RandomForestRegressor(max_depth=2, n_estimators=10),
                                     LinearRegression(),
                                     Lasso(alpha=0.1)])
@pytest.mark.parametrize('inf_model', ['DML2018'])
@pytest.mark.parametrize('dml_procedure', ['dml1', 'dml2'])
def test_dml_pliv(generate_data_iv, idx, learner, inf_model, dml_procedure):
    resampling = KFold(n_splits=2, shuffle=True)
    
    # Set machine learning methods for m & g
    ml_learners = {'ml_m': clone(clone(learner)),
                   'ml_g': clone(clone(learner)),
                   'ml_r': clone(clone(learner))}
    
    dml_pliv_obj = DoubleMLPLIV(resampling,
                                ml_learners,
                                dml_procedure,
                                inf_model)
    data = generate_data_iv[idx]
    np.random.seed(3141)
    dml_pliv_obj.fit(data.loc[:, data.columns.str.startswith('X')].values,
                    data['y'].values, data['d'].values, data['z'].values)
    
    np.random.seed(3141)
    y = data['y'].values
    X = data.loc[:, data.columns.str.startswith('X')].values
    d = data['d'].values
    z = data['z'].values
    
    smpls = [(train, test) for train, test in resampling.split(X)]
    
    g_hat, m_hat, r_hat = fit_nuisance_pliv(y, X, d, z,
                                            clone(learner), clone(learner), clone(learner),
                                            smpls)
    
    if dml_procedure == 'dml1':
        res_manual, se_manual = pliv_dml1(y, X, d,
                                          z,
                                          g_hat, m_hat, r_hat,
                                          smpls, inf_model)
    elif dml_procedure == 'dml2':
        res_manual, se_manual = pliv_dml2(y, X, d,
                                          z,
                                          g_hat, m_hat, r_hat,
                                          smpls, inf_model)
    
    assert math.isclose(dml_pliv_obj.coef_[0], res_manual, rel_tol=1e-9, abs_tol=1e-4)
    assert math.isclose(dml_pliv_obj.se_[0], se_manual, rel_tol=1e-9, abs_tol=1e-4)
    
    for bootstrap in ['Bayes', 'normal', 'wild']:
        np.random.seed(3141)
        boot_theta = boot_pliv(res_manual,
                               y, d,
                               z,
                               g_hat, m_hat, r_hat,
                               smpls, inf_model,
                               se_manual,
                               bootstrap, 500)
        
        np.random.seed(3141)
        dml_pliv_obj.bootstrap(method = bootstrap, n_rep=500)
        assert np.allclose(dml_pliv_obj.boot_coef_, boot_theta, rtol=1e-9, atol=1e-4)
    
    return
    
