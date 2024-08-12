import pytest

from doubleml.irm.irm import DoubleMLIRM
from doubleml.double_ml_framework import concat

from sklearn.linear_model import LinearRegression, LogisticRegression


@pytest.fixture(scope='module',
                params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope='module')
def dml_framework_sensitivity_fixture(n_rep, generate_data_simple):
    dml_data, dml_data_2 = generate_data_simple

    ml_g = LinearRegression()
    ml_m = LogisticRegression()

    dml_irm_obj = DoubleMLIRM(dml_data, ml_g, ml_m, n_rep=n_rep)
    dml_irm_obj.fit()
    dml_framework_obj = dml_irm_obj.construct_framework()
    dml_framework_obj.sensitivity_analysis()

    # add objects
    dml_framework_obj_add_obj = dml_framework_obj + dml_framework_obj

    # substract objects
    dml_irm_obj_2 = DoubleMLIRM(dml_data_2, ml_g, ml_m, n_rep=n_rep)
    dml_irm_obj_2.fit()
    dml_framework_obj_2 = dml_irm_obj_2.construct_framework()
    dml_framework_obj_sub_obj = dml_framework_obj - dml_framework_obj_2

    # multiply objects
    dml_framework_obj_mul_obj = dml_framework_obj * 2

    # concat objects
    dml_framework_obj_concat = concat([dml_framework_obj, dml_framework_obj])

    result_dict = {
        'dml_obj': dml_irm_obj,
        'dml_obj_2': dml_irm_obj_2,
        'dml_framework_obj': dml_framework_obj,
        'dml_framework_obj_2': dml_framework_obj_2,
        'dml_framework_obj_add_obj': dml_framework_obj_add_obj,
        'dml_framework_obj_sub_obj': dml_framework_obj_sub_obj,
        'dml_framework_obj_mul_obj': dml_framework_obj_mul_obj,
        'dml_framework_obj_concat': dml_framework_obj_concat,
        'n_rep': n_rep,
    }
    return result_dict


@pytest.mark.ci
def test_dml_framework_sensitivity_shapes(dml_framework_sensitivity_fixture):
    n_rep = dml_framework_sensitivity_fixture['dml_framework_obj'].n_rep
    n_thetas = dml_framework_sensitivity_fixture['dml_framework_obj'].n_thetas
    n_obs = dml_framework_sensitivity_fixture['dml_framework_obj'].n_obs

    object_list = ['dml_framework_obj',
                   'dml_framework_obj_2',
                   'dml_framework_obj_add_obj',
                   'dml_framework_obj_sub_obj',
                   'dml_framework_obj_mul_obj']
    var_keys = ['sigma2', 'nu2']
    score_keys = ['psi_sigma2', 'psi_nu2', 'riesz_rep']

    for obj in object_list:
        assert dml_framework_sensitivity_fixture[obj]._sensitivity_implemented
        for key in var_keys:
            assert dml_framework_sensitivity_fixture[obj]._sensitivity_elements[key].shape == \
                (1, n_thetas, n_rep)
        for key in score_keys:
            assert dml_framework_sensitivity_fixture[obj]._sensitivity_elements[key].shape == \
                (n_obs, n_thetas, n_rep)

    # separate test for concat
    for key in var_keys:
        assert dml_framework_sensitivity_fixture['dml_framework_obj_concat']._sensitivity_elements[key].shape == \
            (1, 2, n_rep)
    for key in score_keys:
        assert dml_framework_sensitivity_fixture['dml_framework_obj_concat']._sensitivity_elements[key].shape == \
            (n_obs, 2, n_rep)


@pytest.mark.ci
def test_dml_framework_sensitivity_summary(dml_framework_sensitivity_fixture):
    # summary without sensitivity analysis
    sensitivity_summary = dml_framework_sensitivity_fixture['dml_framework_obj_2'].sensitivity_summary
    substring = 'Apply sensitivity_analysis() to generate sensitivity_summary.'
    assert substring in sensitivity_summary

    # summary with sensitivity analysis
    sensitivity_summary = dml_framework_sensitivity_fixture['dml_framework_obj'].sensitivity_summary
    assert isinstance(sensitivity_summary, str)
    substrings = [
        '\n------------------ Scenario          ------------------\n',
        '\n------------------ Bounds with CI    ------------------\n',
        '\n------------------ Robustness Values ------------------\n',
        'Significance Level: level=',
        'Sensitivity parameters: cf_y='
    ]
    for substring in substrings:
        assert substring in sensitivity_summary
