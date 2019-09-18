import numpy as np
import pytest
import math
import scipy

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from dml.double_ml_iivm import DoubleMLPIIVM

from dml.tests.helper_general import get_n_datasets
from dml.tests.helper_iivm_manual import iivm_dml1, iivm_dml2, fit_nuisance_iivm, boot_iivm


# number of datasets per dgp
n_datasets = get_n_datasets()

@pytest.mark.parametrize('idx', range(n_datasets))
@pytest.mark.parametrize('learner', [[LogisticRegression(solver='lbfgs', max_iter=250),
                                      LinearRegression()],
                                     [RandomForestClassifier(max_depth=2, n_estimators=10),
                                      RandomForestRegressor(max_depth=2, n_estimators=10)]])
@pytest.mark.parametrize('inf_model', ['LATE'])
@pytest.mark.parametrize('dml_procedure', ['dml1', 'dml2'])
def test_dml_iivm(generate_data_iivm, idx, learner, inf_model, dml_procedure):
    resampling = KFold(n_splits=2, shuffle=True)
    
    # Set machine learning methods for m & g
    ml_learners = {'ml_m': clone(clone(learner[0])),
                   'ml_g': clone(clone(learner[1])),
                   'ml_r': clone(clone(learner[0]))}
    
    dml_iivm_obj = DoubleMLPIIVM(resampling,
                                 ml_learners,
                                 dml_procedure,
                                 inf_model)
    data = generate_data_iivm[idx]
    np.random.seed(3141)
    dml_iivm_obj.fit(data['X'], data['y'], data['d'], data['z'])
    
    np.random.seed(3141)
    smpls = [(train, test) for train, test in resampling.split(data['X'])]
    
    g_hat0, g_hat1, m_hat, r_hat0, r_hat1 = fit_nuisance_iivm(data['y'], data['X'], data['d'], data['z'],
                                                              clone(learner[0]), clone(learner[1]), clone(learner[0]), smpls)
    
    
    if dml_procedure == 'dml1':
        res_manual, se_manual = iivm_dml1(data['y'], data['X'], data['d'], data['z'],
                                         g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                                         smpls, inf_model)
    elif dml_procedure == 'dml2':
        res_manual, se_manual = iivm_dml2(data['y'], data['X'], data['d'], data['z'],
                                         g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                                         smpls, inf_model)
    
    assert math.isclose(dml_iivm_obj.coef_, res_manual, rel_tol=1e-9, abs_tol=1e-4)
    #assert math.isclose(dml_iivm_obj.se_, se_manual, rel_tol=1e-9, abs_tol=1e-4)
    
    for bootstrap in ['normal']:
        np.random.seed(3141)
        boot_theta = boot_iivm(res_manual,
                              data['y'], data['d'], data['z'],
                              g_hat0, g_hat1, m_hat, r_hat0, r_hat1,
                              smpls, inf_model,
                              se_manual,
                              bootstrap, 500)
        
        np.random.seed(3141)
        dml_iivm_obj.bootstrap(method = bootstrap, n_rep=500)
        assert np.allclose(dml_iivm_obj.boot_coef_, boot_theta, rtol=1e-9, atol=1e-4)
    
    return

