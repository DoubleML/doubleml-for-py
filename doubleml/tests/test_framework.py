import pytest
import numpy as np
import pandas as pd

from doubleml.double_ml_base_linear import DoubleMLBaseLinear
from doubleml.double_ml_framework import DoubleMLFramework


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module',
                params=[1, 5])
def n_thetas(request):
    return request.param


@pytest.fixture(scope='module')
def dml_framework_fixture(n_rep, n_thetas):
    n_obs = 100
    psi_elements = {
        'psi_a': np.ones(shape=(n_obs, n_thetas, n_rep)),
        'psi_b': np.random.normal(size=(n_obs, n_thetas, n_rep)),
    }
    dml_obj = DoubleMLBaseLinear(psi_elements).estimate_thetas()
    dml_framework_obj = DoubleMLFramework(dml_obj)

    ci = dml_framework_obj.confint(joint=False, level=0.95)
    dml_framework_obj.bootstrap(method='normal')
    ci_joint = dml_framework_obj.confint(joint=True, level=0.95)

    # add objects
    dml_framework_obj_add_obj = dml_framework_obj + dml_framework_obj
    ci_add_obj = dml_framework_obj_add_obj.confint(joint=False, level=0.95)
    dml_framework_obj_add_obj.bootstrap(method='normal')
    ci_joint_add_obj = dml_framework_obj_add_obj.confint(joint=True, level=0.95)

    # substract objects
    psi_elements_2 = {
        'psi_a': np.ones(shape=(n_obs, n_thetas, n_rep)),
        'psi_b': np.random.normal(size=(n_obs, n_thetas, n_rep)) + 1.0,
    }
    dml_obj_2 = DoubleMLBaseLinear(psi_elements_2).estimate_thetas()
    dml_framework_obj_2 = DoubleMLFramework(dml_obj_2)
    dml_framework_obj_sub_obj = dml_framework_obj - dml_framework_obj_2
    ci_sub_obj = dml_framework_obj_sub_obj.confint(joint=False, level=0.95)
    dml_framework_obj_sub_obj.bootstrap(method='normal')
    ci_joint_sub_obj = dml_framework_obj_sub_obj.confint(joint=True, level=0.95)

    # multiply objects
    dml_framework_obj_mul_obj = dml_framework_obj * 2
    ci_mul_obj = dml_framework_obj_mul_obj.confint(joint=False, level=0.95)
    dml_framework_obj_mul_obj.bootstrap(method='normal')
    ci_joint_mul_obj = dml_framework_obj_mul_obj.confint(joint=True, level=0.95)

    result_dict = {
        'dml_obj': dml_obj,
        'dml_obj_2': dml_obj_2,
        'dml_framework_obj': dml_framework_obj,
        'dml_framework_obj_add_obj': dml_framework_obj_add_obj,
        'dml_framework_obj_sub_obj': dml_framework_obj_sub_obj,
        'dml_framework_obj_mul_obj': dml_framework_obj_mul_obj,
        'ci': ci,
        'ci_add_obj': ci_add_obj,
        'ci_sub_obj': ci_sub_obj,
        'ci_mul_obj': ci_mul_obj,
        'ci_joint': ci_joint,
        'ci_joint_add_obj': ci_joint_add_obj,
        'ci_joint_sub_obj': ci_joint_sub_obj,
        'ci_joint_mul_obj': ci_joint_mul_obj,
    }
    return result_dict


@pytest.mark.rewrite
@pytest.mark.ci
def test_dml_framework_theta(dml_framework_fixture):
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj'].all_thetas,
        dml_framework_fixture['dml_obj'].all_thetas
    )
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_add_obj'].all_thetas,
        dml_framework_fixture['dml_obj'].all_thetas + dml_framework_fixture['dml_obj'].all_thetas
    )
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_sub_obj'].all_thetas,
        dml_framework_fixture['dml_obj'].all_thetas - dml_framework_fixture['dml_obj_2'].all_thetas
    )
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_mul_obj'].all_thetas,
        2*dml_framework_fixture['dml_obj'].all_thetas
    )


@pytest.mark.rewrite
@pytest.mark.ci
def test_dml_framework_se(dml_framework_fixture):
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj'].all_ses,
        dml_framework_fixture['dml_obj'].all_ses
    )
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_add_obj'].all_ses,
        2*dml_framework_fixture['dml_obj'].all_ses
    )
    scaling = dml_framework_fixture['dml_obj'].var_scaling_factors.reshape(-1, 1)
    sub_var = np.mean(
        np.square(dml_framework_fixture['dml_obj'].psi - dml_framework_fixture['dml_obj_2'].psi),
        axis=0)
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_sub_obj'].all_ses,
        np.sqrt(sub_var / scaling)
    )
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_mul_obj'].all_ses,
        2*dml_framework_fixture['dml_obj'].all_ses
    )


@pytest.mark.rewrite
@pytest.mark.ci
def test_dml_framework_ci(dml_framework_fixture):
    assert isinstance(dml_framework_fixture['ci'], pd.DataFrame)
    assert isinstance(dml_framework_fixture['ci_joint'], pd.DataFrame)
    assert isinstance(dml_framework_fixture['ci_add_obj'], pd.DataFrame)
    assert isinstance(dml_framework_fixture['ci_joint_add_obj'], pd.DataFrame)
    assert isinstance(dml_framework_fixture['ci_sub_obj'], pd.DataFrame)
    assert isinstance(dml_framework_fixture['ci_joint_sub_obj'], pd.DataFrame)
    assert isinstance(dml_framework_fixture['ci_mul_obj'], pd.DataFrame)
    assert isinstance(dml_framework_fixture['ci_joint_mul_obj'], pd.DataFrame)
