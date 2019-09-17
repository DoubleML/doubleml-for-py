import numpy as np
import pytest
import math
import scipy

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from dml.double_ml_irm import DoubleMLPIRM

from dml.tests.helper_general import get_n_datasets
from dml.tests.helper_irm_manual import irm_dml1, irm_dml2, fit_nuisance_irm, boot_irm


# number of datasets per dgp
n_datasets = get_n_datasets()

@pytest.mark.parametrize('idx', range(n_datasets))
@pytest.mark.parametrize('learner', [[LogisticRegression(solver='lbfgs', max_iter=250),
                                      LinearRegression()],
                                     [RandomForestClassifier(max_depth=2, n_estimators=10),
                                      RandomForestRegressor(max_depth=2, n_estimators=10)]])
@pytest.mark.parametrize('inf_model', ['ATE', 'ATTE'])
@pytest.mark.parametrize('dml_procedure', ['dml1', 'dml2'])
def test_dml_irm(generate_data_irm, idx, learner, inf_model, dml_procedure):
    resampling = KFold(n_splits=2, shuffle=True)
    
    # Set machine learning methods for m & g
    ml_learners = {'ml_m': clone(clone(learner[0])),
                   'ml_g': clone(clone(learner[1]))}
    
    dml_irm_obj = DoubleMLPIRM(resampling,
                               ml_learners,
                               dml_procedure,
                               inf_model)
    data = generate_data_irm[idx]
    np.random.seed(3141)
    dml_irm_obj.fit(data['X'], data['y'], data['d'])
    
    np.random.seed(3141)
    smpls = [(train, test) for train, test in resampling.split(data['X'])]
    
    g_hat0, g_hat1, m_hat = fit_nuisance_irm(data['y'], data['X'], data['d'],
                                             clone(learner[0]), clone(learner[1]), smpls,
                                             inf_model)
    
    if dml_procedure == 'dml1':
        res_manual, se_manual = irm_dml1(data['y'], data['X'], data['d'],
                                         g_hat0, g_hat1, m_hat,
                                         smpls, inf_model)
    elif dml_procedure == 'dml2':
        res_manual, se_manual = irm_dml2(data['y'], data['X'], data['d'],
                                         g_hat0, g_hat1, m_hat,
                                         smpls, inf_model)
    
    assert math.isclose(dml_irm_obj.coef_, res_manual, rel_tol=1e-9, abs_tol=1e-4)
    #assert math.isclose(dml_irm_obj.se_[0], se_manual, rel_tol=1e-9, abs_tol=1e-4)
    #
    #for bootstrap in ['normal']:
    #    np.random.seed(3141)
    #    boot_theta = boot_irm(res_manual,
    #                          data['y'], data['d'],
    #                          g_hat, m_hat,
    #                          smpls, inf_model,
    #                          se_manual,
    #                          bootstrap, 500)
    #    
    #    np.random.seed(3141)
    #    dml_irm_obj.bootstrap(method = bootstrap, n_rep=500)
    #    assert np.allclose(dml_irm_obj.boot_coef_, boot_theta, rtol=1e-9, atol=1e-4)
    
    return
    
