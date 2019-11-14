import numpy as np
import pytest
import math

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LogisticRegression, LinearRegression

from dml.double_ml_data import DoubleMLData
from dml.double_ml_iivm import DoubleMLIIVM

from dml.tests.helper_general import get_n_datasets

from rpy2.robjects import pandas2ri
from dml.tests.helper_pyvsr import r_IIVM

# number of datasets per dgp
n_datasets = get_n_datasets()

@pytest.fixture(scope='module',
                params = range(n_datasets))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['LATE'])
def inf_model(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope="module")
def dml_iivm_pyvsr_fixture(generate_data_iivm, idx, inf_model, dml_procedure):
    boot_methods = ['normal']
    
    resampling = KFold(n_splits=2, shuffle=False)
    
    # Set machine learning methods for m & gg
    learner_classif = LogisticRegression(penalty='none', solver='newton-cg')
    learner_reg = LinearRegression()
    ml_learners = {'ml_m': clone(learner_classif),
                   'ml_g': clone(learner_reg),
                   'ml_r': clone(learner_classif)}
    
    dml_iivm_obj = DoubleMLIIVM(resampling,
                                ml_learners,
                                dml_procedure,
                                inf_model)
    data = generate_data_iivm[idx]
    np.random.seed(3141)
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()
    obj_dml_data = DoubleMLData(data, X_cols, 'y', ['d'], 'z')
    dml_iivm_obj.fit(obj_dml_data)

    # fit the DML model in R
    r_dataframe = pandas2ri.py2rpy(data)
    res_r = r_IIVM(r_dataframe, inf_model, dml_procedure)

    res_dict = {'coef_py': dml_iivm_obj.coef,
                'coef_r': res_r[0],
                'se_py': dml_iivm_obj.se,
                'se_r': res_r[1]}

    return res_dict


def test_dml_iivm_pyvsr_coef(dml_iivm_pyvsr_fixture):
    assert math.isclose(dml_iivm_pyvsr_fixture['coef_py'],
                        dml_iivm_pyvsr_fixture['coef_r'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_iivm_pyvsr_se(dml_iivm_pyvsr_fixture):
    assert math.isclose(dml_iivm_pyvsr_fixture['se_py'],
                        dml_iivm_pyvsr_fixture['se_r'],
                        rel_tol=1e-9, abs_tol=1e-4)

