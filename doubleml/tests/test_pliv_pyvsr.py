import numpy as np
import pytest
import math

from sklearn.base import clone
from sklearn.linear_model import LinearRegression

import doubleml as dml

from doubleml.tests.helper_general import get_n_datasets
from doubleml.tests.helper_pyvsr import export_smpl_split_to_r, r_MLPLIV, \
    r_MLPLIV_PARTIAL_X, r_MLPLIV_PARTIAL_Z, r_MLPLIV_PARTIAL_XZ

rpy2 = pytest.importorskip("rpy2")
from rpy2.robjects import pandas2ri
pandas2ri.activate()


# number of datasets per dgp
n_datasets = get_n_datasets()

@pytest.fixture(scope='module',
                params = range(n_datasets))
def idx(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['partialling out'])
def score(request):
    return request.param


@pytest.fixture(scope='module',
                params = ['dml1', 'dml2'])
def dml_procedure(request):
    return request.param


@pytest.fixture(scope='module')
def dml_pliv_pyvsr_fixture(generate_data_iv, idx, score, dml_procedure):
    n_folds = 2

    # collect data
    data = generate_data_iv[idx]
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()
    
    # Set machine learning methods for g, m & r
    learner = LinearRegression()
    ml_g = clone(learner)
    ml_m = clone(learner)
    ml_r = clone(learner)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], X_cols, 'Z1')
    dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data,
                                    ml_g, ml_m, ml_r,
                                    n_folds,
                                    dml_procedure=dml_procedure)

    dml_pliv_obj.fit()


    # fit the DML model in R
    all_train, all_test = export_smpl_split_to_r(dml_pliv_obj.smpls[0])

    r_dataframe = pandas2ri.py2rpy(data)
    res_r = r_MLPLIV(r_dataframe, 'partialling out', dml_procedure,
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


@pytest.fixture(scope='module')
def dml_pliv_partial_x_pyvsr_fixture(generate_data_pliv_partialX, idx, score, dml_procedure):
    n_folds = 2

    # collect data
    data = generate_data_pliv_partialX[idx].data
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()
    Z_cols = data.columns[data.columns.str.startswith('Z')].tolist()
    X_cols = X_cols[0:100]  # no real ml-learner in UT, so keep dim low
    data = data[X_cols + Z_cols + ['d', 'y']]

    # Set machine learning methods for g, m & r
    learner = LinearRegression()
    ml_g = clone(learner)
    ml_m = clone(learner)
    ml_r = clone(learner)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], X_cols, Z_cols)
    dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data,
                                    ml_g, ml_m, ml_r,
                                    n_folds,
                                    dml_procedure=dml_procedure)

    dml_pliv_obj.fit()


    # fit the DML model in R
    all_train, all_test = export_smpl_split_to_r(dml_pliv_obj.smpls[0])

    r_dataframe = pandas2ri.py2rpy(data)
    res_r = r_MLPLIV_PARTIAL_X(r_dataframe, 'partialling out', dml_procedure,
                              all_train, all_test)
    print(res_r)

    res_dict = {'coef_py': dml_pliv_obj.coef,
                'coef_r': res_r[0],
                'se_py': dml_pliv_obj.se,
                'se_r': res_r[1]}

    return res_dict


def test_dml_pliv_partial_x_pyvsr_coef(dml_pliv_partial_x_pyvsr_fixture):
    assert math.isclose(dml_pliv_partial_x_pyvsr_fixture['coef_py'],
                        dml_pliv_partial_x_pyvsr_fixture['coef_r'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_partial_x_pyvsr_se(dml_pliv_partial_x_pyvsr_fixture):
    assert math.isclose(dml_pliv_partial_x_pyvsr_fixture['se_py'],
                        dml_pliv_partial_x_pyvsr_fixture['se_r'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.fixture(scope='module')
def dml_pliv_partial_z_pyvsr_fixture(generate_data_pliv_partialZ, idx, score, dml_procedure):
    n_folds = 2

    # collect data
    data = generate_data_pliv_partialZ[idx]
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()
    Z_cols = data.columns[data.columns.str.startswith('Z')].tolist()
    Z_cols = Z_cols[0:50]  # no real ml-learner in UT, so keep dim low
    data = data[X_cols + Z_cols + ['d', 'y']]

    # Set machine learning methods for g, m & r
    learner = LinearRegression()
    ml_r = clone(learner)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], X_cols, Z_cols)
    dml_pliv_obj = dml.DoubleMLPLIV._partialZ(obj_dml_data,
                                              ml_r,
                                              n_folds,
                                              dml_procedure=dml_procedure)

    dml_pliv_obj.fit()

    # fit the DML model in R
    all_train, all_test = export_smpl_split_to_r(dml_pliv_obj.smpls[0])

    r_dataframe = pandas2ri.py2rpy(data)
    res_r = r_MLPLIV_PARTIAL_Z(r_dataframe, 'partialling out', dml_procedure,
                     all_train, all_test)
    print(res_r)

    res_dict = {'coef_py': dml_pliv_obj.coef,
                'coef_r': res_r[0],
                'se_py': dml_pliv_obj.se,
                'se_r': res_r[1]}

    return res_dict


def test_dml_pliv_partial_z_pyvsr_coef(dml_pliv_partial_z_pyvsr_fixture):
    assert math.isclose(dml_pliv_partial_z_pyvsr_fixture['coef_py'],
                        dml_pliv_partial_z_pyvsr_fixture['coef_r'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_partial_z_pyvsr_se(dml_pliv_partial_z_pyvsr_fixture):
    assert math.isclose(dml_pliv_partial_z_pyvsr_fixture['se_py'],
                        dml_pliv_partial_z_pyvsr_fixture['se_r'],
                        rel_tol=1e-9, abs_tol=1e-4)


@pytest.fixture(scope='module')
def dml_pliv_partial_xz_pyvsr_fixture(generate_data_pliv_partialXZ, idx, score, dml_procedure):
    n_folds = 2

    # collect data
    data = generate_data_pliv_partialXZ[idx].data
    X_cols = data.columns[data.columns.str.startswith('X')].tolist()
    Z_cols = data.columns[data.columns.str.startswith('Z')].tolist()
    X_cols = X_cols[0:100]  # no real ml-learner in UT, so keep dim low
    Z_cols = Z_cols[0:50]  # no real ml-learner in UT, so keep dim low
    data = data[X_cols + Z_cols + ['d', 'y']]

    # Set machine learning methods for g, m & r
    learner = LinearRegression()
    ml_g = clone(learner)
    ml_m = clone(learner)
    ml_r = clone(learner)

    np.random.seed(3141)
    obj_dml_data = dml.DoubleMLData(data, 'y', ['d'], X_cols, Z_cols)
    dml_pliv_obj = dml.DoubleMLPLIV._partialXZ(obj_dml_data,
                                               ml_g, ml_m, ml_r,
                                               n_folds,
                                               dml_procedure=dml_procedure)

    dml_pliv_obj.fit()

    # fit the DML model in R
    all_train, all_test = export_smpl_split_to_r(dml_pliv_obj.smpls[0])

    r_dataframe = pandas2ri.py2rpy(data)
    res_r = r_MLPLIV_PARTIAL_XZ(r_dataframe, 'partialling out', dml_procedure,
                     all_train, all_test)
    print(res_r)

    res_dict = {'coef_py': dml_pliv_obj.coef,
                'coef_r': res_r[0],
                'se_py': dml_pliv_obj.se,
                'se_r': res_r[1]}

    return res_dict


def test_dml_pliv_partial_xz_pyvsr_coef(dml_pliv_partial_xz_pyvsr_fixture):
    assert math.isclose(dml_pliv_partial_xz_pyvsr_fixture['coef_py'],
                        dml_pliv_partial_xz_pyvsr_fixture['coef_r'],
                        rel_tol=1e-9, abs_tol=1e-4)


def test_dml_pliv_partial_xz_pyvsr_se(dml_pliv_partial_xz_pyvsr_fixture):
    assert math.isclose(dml_pliv_partial_xz_pyvsr_fixture['se_py'],
                        dml_pliv_partial_xz_pyvsr_fixture['se_r'],
                        rel_tol=1e-9, abs_tol=1e-4)
