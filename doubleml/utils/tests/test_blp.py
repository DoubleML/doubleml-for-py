import numpy as np
import pandas as pd
import pytest
import copy

import doubleml as dml
from ._utils_blp_manual import fit_blp, blp_confint


@pytest.fixture(scope='module',
                params=[True, False])
def ci_joint(request):
    return request.param


@pytest.fixture(scope='module',
                params=[0.95, 0.9])
def ci_level(request):
    return request.param


@pytest.fixture(scope='module',
                params=["nonrobust", "HC0", "HC1", "HC2", "HC3"])
def cov_type(request):
    return request.param


@pytest.fixture(scope='module',
                params=[True, False])
def use_t(request):
    return request.param


@pytest.fixture(scope='module')
def dml_blp_fixture(ci_joint, ci_level, cov_type, use_t):
    n = 50
    kwargs = {'cov_type': cov_type, 'use_t': use_t}

    np.random.seed(42)
    random_basis = pd.DataFrame(np.random.normal(0, 1, size=(n, 3)))
    random_signal = np.random.normal(0, 1, size=(n, ))

    blp = dml.DoubleMLBLP(random_signal, random_basis)

    blp_obj = copy.copy(blp)
    blp.fit(**kwargs)
    blp_manual = fit_blp(random_signal, random_basis, **kwargs)

    np.random.seed(42)
    ci_1 = blp.confint(random_basis, joint=ci_joint, level=ci_level, n_rep_boot=1000)
    np.random.seed(42)
    ci_2 = blp.confint(joint=ci_joint, level=ci_level, n_rep_boot=1000)
    expected_ci_2 = np.vstack((
        blp.blp_model.conf_int(alpha=(1-ci_level)/2)[0],
        blp.blp_model.params,
        blp.blp_model.conf_int(alpha=(1-ci_level)/2)[1])).T

    np.random.seed(42)
    ci_manual = blp_confint(blp_manual, random_basis, joint=ci_joint, level=ci_level, n_rep_boot=1000)

    res_dict = {'coef': blp.blp_model.params,
                'coef_manual': blp_manual.params,
                'values': blp.blp_model.fittedvalues,
                'values_manual':  blp_manual.fittedvalues,
                'omega': blp.blp_omega,
                'omega_manual': blp_manual.cov_params().to_numpy(),
                'basis': blp.basis,
                'signal': blp.orth_signal,
                'ci_1': ci_1,
                'ci_2': ci_2,
                'expected_ci_2': expected_ci_2,
                'ci_manual': ci_manual,
                'blp_model': blp,
                'unfitted_blp_model': blp_obj}

    return res_dict


@pytest.mark.ci
def test_dml_blp_coef(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture['coef'],
                       dml_blp_fixture['coef_manual'],
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_blp_values(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture['values'],
                       dml_blp_fixture['values_manual'],
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_blp_omega(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture['omega'],
                       dml_blp_fixture['omega_manual'],
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_blp_ci_2(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture['expected_ci_2'],
                       dml_blp_fixture['ci_2'],
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_blp_ci_1(dml_blp_fixture):
    assert np.allclose(dml_blp_fixture['ci_1'],
                       dml_blp_fixture['ci_manual'],
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_blp_return_types(dml_blp_fixture):
    assert isinstance(dml_blp_fixture['blp_model'].__str__(), str)
    assert isinstance(dml_blp_fixture['blp_model'].summary, pd.DataFrame)
    assert isinstance(dml_blp_fixture['unfitted_blp_model'].summary, pd.DataFrame)


@pytest.mark.ci
def test_dml_blp_defaults():
    n = 50
    np.random.seed(42)
    random_basis = pd.DataFrame(np.random.normal(0, 1, size=(n, 3)))
    random_signal = np.random.normal(0, 1, size=(n, ))

    blp = dml.DoubleMLBLP(random_signal, random_basis)
    blp.fit()

    assert np.allclose(blp.blp_omega,
                       blp.blp_model.cov_HC0,
                       rtol=1e-9, atol=1e-4)

    assert blp._is_gate is False


@pytest.mark.ci
def test_doubleml_exception_blp():
    random_basis = pd.DataFrame(np.random.normal(0, 1, size=(2, 3)))
    signal = np.array([1, 2])

    msg = "The signal must be of np.ndarray type. Signal of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml.DoubleMLBLP(orth_signal=1, basis=random_basis)
    msg = 'The signal must be of one dimensional. Signal of dimensions 2 was passed.'
    with pytest.raises(ValueError, match=msg):
        dml.DoubleMLBLP(orth_signal=np.array([[1], [2]]), basis=random_basis)
    msg = "The basis must be of DataFrame type. Basis of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml.DoubleMLBLP(orth_signal=signal, basis=1)
    msg = 'Invalid pd.DataFrame: Contains duplicate column names.'
    with pytest.raises(ValueError, match=msg):
        dml.DoubleMLBLP(orth_signal=signal, basis=pd.DataFrame(np.array([[1, 2], [4, 5]]),
                                                               columns=['a_1', 'a_1']))

    dml_blp_confint = dml.DoubleMLBLP(orth_signal=signal, basis=random_basis)
    msg = r'Apply fit\(\) before confint\(\).'
    with pytest.raises(ValueError, match=msg):
        dml_blp_confint.confint(random_basis)

    dml_blp_confint.fit()
    msg = 'joint must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_blp_confint.confint(random_basis, joint=1)
    msg = "The confidence level must be of float type. 5% of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_blp_confint.confint(random_basis, level='5%')
    msg = r'The confidence level must be in \(0,1\). 0.0 was passed.'
    with pytest.raises(ValueError, match=msg):
        dml_blp_confint.confint(random_basis, level=0.)
    msg = "The number of bootstrap replications must be of int type. 500 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_blp_confint.confint(random_basis, n_rep_boot='500')
    msg = 'The number of bootstrap replications must be positive. 0 was passed.'
    with pytest.raises(ValueError, match=msg):
        dml_blp_confint.confint(random_basis, n_rep_boot=0)
    msg = 'Invalid basis: DataFrame has to have the exact same number and ordering of columns.'
    with pytest.raises(ValueError, match=msg):
        dml_blp_confint.confint(basis=pd.DataFrame(np.array([[1], [4]]), columns=['a_1']))
    msg = 'Invalid basis: DataFrame has to have the exact same number and ordering of columns.'
    with pytest.raises(ValueError, match=msg):
        dml_blp_confint.confint(basis=pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=['x_1', 'x_2', 'x_3']))
