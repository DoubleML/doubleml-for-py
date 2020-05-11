import numpy as np
import pytest
import math

from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml

from doubleml.tests.helper_general import get_n_datasets
from doubleml.tests.helper_pyvsr import export_smpl_split_to_r, r_IRM

from rpy2.robjects import pandas2ri
pandas2ri.activate()

# number of datasets per dgp
n_datasets = get_n_datasets()


@pytest.fixture(scope='module',
                params = range(n_datasets))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['ATE', 'ATTE'])
def inf_model(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module')
def dml_irm_pyvsr_fixture(generate_data_irm, idx, inf_model, dml_procedure):
    n_folds = 2

    # collect data
    data = generate_data_irm[idx]
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()

    # Set machine learning methods for m & g
    learner_classif = LogisticRegression(penalty='none', solver='newton-cg')
    learner_reg = LinearRegression()
    ml_learners = {'ml_m': clone(learner_classif),
                   'ml_g': clone(learner_reg)}
    
    dml_irm_obj = dml.DoubleMLIRM(data, X_cols, 'y', ['d'],
                              ml_learners,
                              n_folds,
                              inf_model=inf_model,
                              dml_procedure=dml_procedure)

    np.random.seed(3141)
    dml_irm_obj.fit()

    # fit the DML model in R
    all_train, all_test = export_smpl_split_to_r(dml_irm_obj.smpls[0])

    r_dataframe = pandas2ri.py2rpy(data)
    if inf_model == 'ATTE':
        inf_model = 'ATET'
    res_r = r_IRM(r_dataframe, inf_model, dml_procedure,
                  all_train, all_test)

    res_dict = {'coef_py': dml_irm_obj.coef,
                'coef_r': res_r[0],
                'se_py': dml_irm_obj.se,
                'se_r': res_r[1]}

    return res_dict


def test_dml_irm_pyvsr_coef(dml_irm_pyvsr_fixture):
    assert math.isclose(dml_irm_pyvsr_fixture['coef_py'],
                        dml_irm_pyvsr_fixture['coef_r'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_irm_pyvsr_se(dml_irm_pyvsr_fixture):
    assert math.isclose(dml_irm_pyvsr_fixture['se_py'],
                        dml_irm_pyvsr_fixture['se_r'],
                        rel_tol=1e-9, abs_tol=1e-4)

