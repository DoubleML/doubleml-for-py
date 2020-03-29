import numpy as np
import pytest
import math

from sklearn.model_selection import KFold
from sklearn.base import clone

from sklearn.linear_model import LinearRegression

from dml.double_ml_data import DoubleMLData
from dml.double_ml_pliv import DoubleMLPLIV

from dml.tests.helper_general import get_n_datasets

from rpy2.robjects import pandas2ri
from dml.tests.helper_pyvsr import export_smpl_split_to_r, r_MLPLIV
pandas2ri.activate()


# number of datasets per dgp
n_datasets = get_n_datasets()

@pytest.fixture(scope='module',
                params = range(n_datasets))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['DML2018'])
def inf_model(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module')
def dml_pliv_pyvsr_fixture(generate_data_iv, idx, inf_model, dml_procedure):
    n_folds = 2

    # collect data
    data = generate_data_iv[idx]
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()
    
    # Set machine learning methods for m & g
    learner = LinearRegression()
    ml_learners = {'ml_m': clone(learner),
                   'ml_g': clone(learner),
                   'ml_r': clone(learner)}

    np.random.seed(3141)
    dml_pliv_obj = DoubleMLPLIV(data, X_cols, 'y', ['d'], 'z',
                                n_folds,
                                ml_learners,
                                dml_procedure,
                                inf_model)

    dml_pliv_obj.fit()


    # fit the DML model in R
    all_train, all_test = export_smpl_split_to_r(dml_pliv_obj.smpls[0])

    r_dataframe = pandas2ri.py2rpy(data)
    assert inf_model == 'DML2018'
    res_r = r_MLPLIV(r_dataframe, 'partialling-out', dml_procedure,
                     all_train, all_test)
    print(res_r)

    res_dict = {'coef_py': dml_pliv_obj.coef,
                'coef_r': res_r[0],
                'se_py': dml_pliv_obj.se,
                'se_r': res_r[1]}
    
    return res_dict


def test_dml_pliv_pyvsr_coef(dml_pliv_pyvsr_fixture):
    assert math.isclose(dml_pliv_pyvsr_fixture['coef_py'],
                        dml_pliv_pyvsr_fixture['coef_r'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_pyvsr_se(dml_pliv_pyvsr_fixture):
    assert math.isclose(dml_pliv_pyvsr_fixture['se_py'],
                        dml_pliv_pyvsr_fixture['se_r'],
                        rel_tol=1e-9, abs_tol=1e-4)

