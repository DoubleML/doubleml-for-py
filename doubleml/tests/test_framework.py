import pytest
import numpy as np
import pandas as pd

from doubleml.datasets import make_irm_data
from doubleml.irm.irm import DoubleMLIRM
from doubleml.double_ml_framework import DoubleMLFramework, concat
from ._utils import generate_dml_dict

from sklearn.linear_model import LinearRegression, LogisticRegression


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

    # generate score samples
    psi_a = np.ones(shape=(n_obs, n_thetas, n_rep))
    psi_b = np.random.normal(size=(n_obs, n_thetas, n_rep))
    doubleml_dict = generate_dml_dict(psi_a, psi_b)
    dml_framework_obj = DoubleMLFramework(doubleml_dict)

    ci = dml_framework_obj.confint(joint=False, level=0.95)
    dml_framework_obj.bootstrap(method='normal')
    ci_joint = dml_framework_obj.confint(joint=True, level=0.95)

    # add objects
    dml_framework_obj_add_obj = dml_framework_obj + dml_framework_obj
    ci_add_obj = dml_framework_obj_add_obj.confint(joint=False, level=0.95)
    dml_framework_obj_add_obj.bootstrap(method='normal')
    ci_joint_add_obj = dml_framework_obj_add_obj.confint(joint=True, level=0.95)

    # substract objects
    psi_a_2 = np.ones(shape=(n_obs, n_thetas, n_rep))
    psi_b_2 = np.random.normal(size=(n_obs, n_thetas, n_rep)) + 1.0
    doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
    dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
    dml_framework_obj_sub_obj = dml_framework_obj - dml_framework_obj_2
    ci_sub_obj = dml_framework_obj_sub_obj.confint(joint=False, level=0.95)
    dml_framework_obj_sub_obj.bootstrap(method='normal')
    ci_joint_sub_obj = dml_framework_obj_sub_obj.confint(joint=True, level=0.95)

    # multiply objects
    dml_framework_obj_mul_obj = dml_framework_obj * 2
    ci_mul_obj = dml_framework_obj_mul_obj.confint(joint=False, level=0.95)
    dml_framework_obj_mul_obj.bootstrap(method='normal')
    ci_joint_mul_obj = dml_framework_obj_mul_obj.confint(joint=True, level=0.95)

    # concat objects
    dml_framework_obj_concat = concat([dml_framework_obj, dml_framework_obj])
    ci_concat = dml_framework_obj_concat.confint(joint=False, level=0.95)
    dml_framework_obj_concat.bootstrap(method='normal')
    ci_joint_concat = dml_framework_obj_concat.confint(joint=True, level=0.95)

    result_dict = {
        'dml_dict': doubleml_dict,
        'dml_dict_2': doubleml_dict_2,
        'dml_framework_obj': dml_framework_obj,
        'dml_framework_obj_add_obj': dml_framework_obj_add_obj,
        'dml_framework_obj_sub_obj': dml_framework_obj_sub_obj,
        'dml_framework_obj_mul_obj': dml_framework_obj_mul_obj,
        'dml_framework_obj_concat': dml_framework_obj_concat,
        'ci': ci,
        'ci_add_obj': ci_add_obj,
        'ci_sub_obj': ci_sub_obj,
        'ci_mul_obj': ci_mul_obj,
        'ci_concat': ci_concat,
        'ci_joint': ci_joint,
        'ci_joint_add_obj': ci_joint_add_obj,
        'ci_joint_sub_obj': ci_joint_sub_obj,
        'ci_joint_mul_obj': ci_joint_mul_obj,
        'ci_joint_concat': ci_joint_concat,
    }
    return result_dict


@pytest.mark.ci
def test_dml_framework_theta(dml_framework_fixture):
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj'].all_thetas,
        dml_framework_fixture['dml_dict']['all_thetas']
    )
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_add_obj'].all_thetas,
        dml_framework_fixture['dml_dict']['all_thetas'] + dml_framework_fixture['dml_dict']['all_thetas']
    )
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_sub_obj'].all_thetas,
        dml_framework_fixture['dml_dict']['all_thetas'] - dml_framework_fixture['dml_dict_2']['all_thetas']
    )
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_mul_obj'].all_thetas,
        2*dml_framework_fixture['dml_dict']['all_thetas']
    )
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_concat'].all_thetas,
        np.vstack((dml_framework_fixture['dml_dict']['all_thetas'], dml_framework_fixture['dml_dict']['all_thetas']))
    )


@pytest.mark.ci
def test_dml_framework_se(dml_framework_fixture):
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj'].all_ses,
        dml_framework_fixture['dml_dict']['all_ses']
    )
    scaling = dml_framework_fixture['dml_dict']['var_scaling_factors'].reshape(-1, 1)
    add_var = np.mean(
        np.square(dml_framework_fixture['dml_dict']['scaled_psi'] + dml_framework_fixture['dml_dict']['scaled_psi']),
        axis=0)
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_add_obj'].all_ses,
        np.sqrt(add_var / scaling)
    )
    scaling = dml_framework_fixture['dml_dict']['var_scaling_factors'].reshape(-1, 1)
    sub_var = np.mean(
        np.square(dml_framework_fixture['dml_dict']['scaled_psi'] - dml_framework_fixture['dml_dict_2']['scaled_psi']),
        axis=0)
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_sub_obj'].all_ses,
        np.sqrt(sub_var / scaling)
    )
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_mul_obj'].all_ses,
        2*dml_framework_fixture['dml_dict']['all_ses']
    )
    assert np.allclose(
        dml_framework_fixture['dml_framework_obj_concat'].all_ses,
        np.vstack((dml_framework_fixture['dml_dict']['all_ses'], dml_framework_fixture['dml_dict']['all_ses']))
    )


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
    assert isinstance(dml_framework_fixture['ci_concat'], pd.DataFrame)
    assert isinstance(dml_framework_fixture['ci_joint_concat'], pd.DataFrame)


@pytest.fixture(scope='module')
def dml_framework_from_doubleml_fixture(n_rep):
    dml_data = make_irm_data()

    ml_g = LinearRegression()
    ml_m = LogisticRegression()

    dml_irm_obj = DoubleMLIRM(dml_data, ml_g, ml_m, n_rep=n_rep)
    dml_irm_obj.fit()
    dml_framework_obj = dml_irm_obj.construct_framework()

    ci = dml_framework_obj.confint(joint=False, level=0.95)
    dml_framework_obj.bootstrap(method='normal')
    ci_joint = dml_framework_obj.confint(joint=True, level=0.95)

    # add objects
    dml_framework_obj_add_obj = dml_framework_obj + dml_framework_obj
    ci_add_obj = dml_framework_obj_add_obj.confint(joint=False, level=0.95)
    dml_framework_obj_add_obj.bootstrap(method='normal')
    ci_joint_add_obj = dml_framework_obj_add_obj.confint(joint=True, level=0.95)

    # substract objects
    dml_data_2 = make_irm_data()
    dml_irm_obj_2 = DoubleMLIRM(dml_data_2, ml_g, ml_m, n_rep=n_rep)
    dml_irm_obj_2.fit()
    dml_framework_obj_2 = dml_irm_obj_2.construct_framework()

    dml_framework_obj_sub_obj = dml_framework_obj - dml_framework_obj_2
    ci_sub_obj = dml_framework_obj_sub_obj.confint(joint=False, level=0.95)
    dml_framework_obj_sub_obj.bootstrap(method='normal')
    ci_joint_sub_obj = dml_framework_obj_sub_obj.confint(joint=True, level=0.95)

    # multiply objects
    dml_framework_obj_mul_obj = dml_framework_obj * 2
    ci_mul_obj = dml_framework_obj_mul_obj.confint(joint=False, level=0.95)
    dml_framework_obj_mul_obj.bootstrap(method='normal')
    ci_joint_mul_obj = dml_framework_obj_mul_obj.confint(joint=True, level=0.95)

    # concat objects
    dml_framework_obj_concat = concat([dml_framework_obj, dml_framework_obj])
    ci_concat = dml_framework_obj_concat.confint(joint=False, level=0.95)
    dml_framework_obj_concat.bootstrap(method='normal')
    ci_joint_concat = dml_framework_obj_concat.confint(joint=True, level=0.95)

    result_dict = {
        'dml_obj': dml_irm_obj,
        'dml_obj_2': dml_irm_obj_2,
        'dml_framework_obj': dml_framework_obj,
        'dml_framework_obj_add_obj': dml_framework_obj_add_obj,
        'dml_framework_obj_sub_obj': dml_framework_obj_sub_obj,
        'dml_framework_obj_mul_obj': dml_framework_obj_mul_obj,
        'dml_framework_obj_concat': dml_framework_obj_concat,
        'ci': ci,
        'ci_add_obj': ci_add_obj,
        'ci_sub_obj': ci_sub_obj,
        'ci_mul_obj': ci_mul_obj,
        'ci_concat': ci_concat,
        'ci_joint': ci_joint,
        'ci_joint_add_obj': ci_joint_add_obj,
        'ci_joint_sub_obj': ci_joint_sub_obj,
        'ci_joint_mul_obj': ci_joint_mul_obj,
        'ci_joint_concat': ci_joint_concat,
        'n_rep': n_rep,
    }
    return result_dict


@pytest.mark.ci
def test_dml_framework_from_doubleml_theta(dml_framework_from_doubleml_fixture):
    assert np.allclose(
        dml_framework_from_doubleml_fixture['dml_framework_obj'].all_thetas,
        dml_framework_from_doubleml_fixture['dml_obj'].all_coef
    )
    assert np.allclose(
        dml_framework_from_doubleml_fixture['dml_framework_obj_add_obj'].all_thetas,
        dml_framework_from_doubleml_fixture['dml_obj'].all_coef + dml_framework_from_doubleml_fixture['dml_obj'].all_coef
    )
    assert np.allclose(
        dml_framework_from_doubleml_fixture['dml_framework_obj_sub_obj'].all_thetas,
        dml_framework_from_doubleml_fixture['dml_obj'].all_coef - dml_framework_from_doubleml_fixture['dml_obj_2'].all_coef
    )
    assert np.allclose(
        dml_framework_from_doubleml_fixture['dml_framework_obj_mul_obj'].all_thetas,
        2*dml_framework_from_doubleml_fixture['dml_obj'].all_coef
    )
    assert np.allclose(
        dml_framework_from_doubleml_fixture['dml_framework_obj_concat'].all_thetas,
        np.vstack((dml_framework_from_doubleml_fixture['dml_obj'].all_coef,
                   dml_framework_from_doubleml_fixture['dml_obj'].all_coef))
    )


@pytest.mark.ci
def test_dml_framework_from_doubleml_se(dml_framework_from_doubleml_fixture):
    assert np.allclose(
        dml_framework_from_doubleml_fixture['dml_framework_obj'].all_ses,
        dml_framework_from_doubleml_fixture['dml_obj'].all_se
    )
    assert np.allclose(
        dml_framework_from_doubleml_fixture['dml_framework_obj_add_obj'].all_ses,
        2*dml_framework_from_doubleml_fixture['dml_obj'].all_se
    )

    if dml_framework_from_doubleml_fixture['n_rep'] == 1:
        # formula only valid for n_rep = 1
        scaling = np.array([dml_framework_from_doubleml_fixture['dml_obj']._var_scaling_factors]).reshape(-1, 1)
        sub_var = np.mean(
            np.square(dml_framework_from_doubleml_fixture['dml_obj'].psi
                      - dml_framework_from_doubleml_fixture['dml_obj_2'].psi),
            axis=0)
        assert np.allclose(
            dml_framework_from_doubleml_fixture['dml_framework_obj_sub_obj'].all_ses,
            np.sqrt(sub_var / scaling)
        )

    assert np.allclose(
        dml_framework_from_doubleml_fixture['dml_framework_obj_mul_obj'].all_ses,
        2*dml_framework_from_doubleml_fixture['dml_obj'].all_se
    )
    assert np.allclose(
        dml_framework_from_doubleml_fixture['dml_framework_obj_concat'].all_ses,
        np.vstack((dml_framework_from_doubleml_fixture['dml_obj'].all_se,
                   dml_framework_from_doubleml_fixture['dml_obj'].all_se))
    )


@pytest.mark.ci
def test_dml_framework_from_doubleml_ci(dml_framework_from_doubleml_fixture):
    assert isinstance(dml_framework_from_doubleml_fixture['ci'], pd.DataFrame)
    assert isinstance(dml_framework_from_doubleml_fixture['ci_joint'], pd.DataFrame)
    assert isinstance(dml_framework_from_doubleml_fixture['ci_add_obj'], pd.DataFrame)
    assert isinstance(dml_framework_from_doubleml_fixture['ci_joint_add_obj'], pd.DataFrame)
    assert isinstance(dml_framework_from_doubleml_fixture['ci_sub_obj'], pd.DataFrame)
    assert isinstance(dml_framework_from_doubleml_fixture['ci_joint_sub_obj'], pd.DataFrame)
    assert isinstance(dml_framework_from_doubleml_fixture['ci_mul_obj'], pd.DataFrame)
    assert isinstance(dml_framework_from_doubleml_fixture['ci_joint_mul_obj'], pd.DataFrame)
    assert isinstance(dml_framework_from_doubleml_fixture['ci_concat'], pd.DataFrame)
    assert isinstance(dml_framework_from_doubleml_fixture['ci_joint_concat'], pd.DataFrame)
