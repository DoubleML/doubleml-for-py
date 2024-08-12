import pytest
import numpy as np
import copy

from doubleml.double_ml_framework import DoubleMLFramework, concat
from ._utils import generate_dml_dict

n_obs = 10
n_thetas = 2
n_rep = 5

# generate score samples
psi_a = np.ones(shape=(n_obs, n_thetas, n_rep))
psi_b = np.random.normal(size=(n_obs, n_thetas, n_rep))
doubleml_dict = generate_dml_dict(psi_a, psi_b)
# add sensitivity elements
doubleml_dict['sensitivity_elements'] = {
    'sigma2': np.ones(shape=(1, n_thetas, n_rep)),
    'nu2': np.ones(shape=(1, n_thetas, n_rep)),
    'psi_sigma2': np.ones(shape=(n_obs, n_thetas, n_rep)),
    'psi_nu2': np.ones(shape=(n_obs, n_thetas, n_rep)),
    'riesz_rep': np.ones(shape=(n_obs, n_thetas, n_rep))
}

# combine objects and estimate parameters
dml_framework_obj_1 = DoubleMLFramework(doubleml_dict)


@pytest.mark.ci
def test_input_exceptions():
    msg = r"The dict must contain the following keys: thetas, ses, all_thetas, all_ses, var_scaling_factors, scaled_psi"
    with pytest.raises(ValueError, match=msg):
        test_dict = {}
        DoubleMLFramework(test_dict)

    msg = r"The shape of thetas does not match the expected shape \(2,\)\."
    with pytest.raises(ValueError, match=msg):
        test_dict = doubleml_dict.copy()
        test_dict['thetas'] = np.ones(shape=(1,))
        DoubleMLFramework(test_dict)

    msg = r"The shape of ses does not match the expected shape \(2,\)\."
    with pytest.raises(ValueError, match=msg):
        test_dict = doubleml_dict.copy()
        test_dict['ses'] = np.ones(shape=(1,))
        DoubleMLFramework(test_dict)

    msg = r"The shape of all_thetas does not match the expected shape \(2, 5\)\."
    with pytest.raises(ValueError, match=msg):
        test_dict = doubleml_dict.copy()
        test_dict['all_thetas'] = np.ones(shape=(1, 5))
        DoubleMLFramework(test_dict)

    msg = r"The shape of all_ses does not match the expected shape \(2, 5\)\."
    with pytest.raises(ValueError, match=msg):
        test_dict = doubleml_dict.copy()
        test_dict['all_ses'] = np.ones(shape=(1, 5))
        DoubleMLFramework(test_dict)

    msg = r"The shape of var_scaling_factors does not match the expected shape \(2,\)\."
    with pytest.raises(ValueError, match=msg):
        test_dict = doubleml_dict.copy()
        test_dict['var_scaling_factors'] = np.ones(shape=(1, 5))
        DoubleMLFramework(test_dict)

    msg = r"The shape of scaled_psi does not match the expected shape \(10, 2, 5\)\."
    with pytest.raises(ValueError, match=msg):
        test_dict = doubleml_dict.copy()
        test_dict['scaled_psi'] = np.ones(shape=(10, 2, 5, 3))
        DoubleMLFramework(test_dict)

    msg = "doubleml_dict must be a dictionary."
    with pytest.raises(TypeError, match=msg):
        DoubleMLFramework(1.0)

    msg = "sensitivity_elements must be a dictionary."
    with pytest.raises(TypeError, match=msg):
        test_dict = doubleml_dict.copy()
        test_dict['sensitivity_elements'] = 1
        DoubleMLFramework(test_dict)

    msg = 'The sensitivity_elements dict must contain the following keys: sigma2, nu2, psi_sigma2, psi_nu2'
    with pytest.raises(ValueError, match=msg):
        test_dict = doubleml_dict.copy()
        test_dict['sensitivity_elements'] = {'sensitivities': np.ones(shape=(n_obs, n_thetas, n_rep))}
        DoubleMLFramework(test_dict)

    msg = r'The shape of sigma2 does not match the expected shape \(1, 2, 5\)\.'
    with pytest.raises(ValueError, match=msg):
        test_dict = copy.deepcopy(doubleml_dict)
        test_dict['sensitivity_elements']['sigma2'] = np.ones(shape=(n_obs, n_rep))
        DoubleMLFramework(test_dict)

    msg = r'The shape of nu2 does not match the expected shape \(1, 2, 5\)\.'
    with pytest.raises(ValueError, match=msg):
        test_dict = copy.deepcopy(doubleml_dict)
        test_dict['sensitivity_elements']['nu2'] = np.ones(shape=(n_obs, n_rep))
        DoubleMLFramework(test_dict)

    msg = r'The shape of psi_sigma2 does not match the expected shape \(10, 2, 5\)\.'
    with pytest.raises(ValueError, match=msg):
        test_dict = copy.deepcopy(doubleml_dict)
        test_dict['sensitivity_elements']['psi_sigma2'] = np.ones(shape=(n_obs, n_thetas, n_rep, 3))
        DoubleMLFramework(test_dict)

    msg = r'The shape of psi_nu2 does not match the expected shape \(10, 2, 5\)\.'
    with pytest.raises(ValueError, match=msg):
        test_dict = copy.deepcopy(doubleml_dict)
        test_dict['sensitivity_elements']['psi_nu2'] = np.ones(shape=(n_obs, n_thetas, n_rep, 3))
        DoubleMLFramework(test_dict)

    msg = r'The shape of riesz_rep does not match the expected shape \(10, 2, 5\)\.'
    with pytest.raises(ValueError, match=msg):
        test_dict = copy.deepcopy(doubleml_dict)
        test_dict['sensitivity_elements']['riesz_rep'] = np.ones(shape=(n_obs, n_thetas, n_rep, 3))
        DoubleMLFramework(test_dict)

    msg = "is_cluster_data has to be boolean. 1.0 of type <class 'float'> was passed."
    with pytest.raises(TypeError, match=msg):
        test_dict = copy.deepcopy(doubleml_dict)
        test_dict['is_cluster_data'] = 1.0
        DoubleMLFramework(test_dict)

    msg = "If is_cluster_data is True, cluster_dict must be provided."
    with pytest.raises(ValueError, match=msg):
        test_dict = copy.deepcopy(doubleml_dict)
        test_dict['is_cluster_data'] = True
        DoubleMLFramework(test_dict)

    msg = "cluster_dict must be a dictionary."
    with pytest.raises(TypeError, match=msg):
        test_dict = copy.deepcopy(doubleml_dict)
        test_dict['is_cluster_data'] = True
        test_dict['cluster_dict'] = 1.0
        DoubleMLFramework(test_dict)

    msg = ('The cluster_dict must contain the following keys: smpls, smpls_cluster,'
           ' cluster_vars, n_folds_per_cluster. Got: cluster_ids.')
    with pytest.raises(ValueError, match=msg):
        test_dict = copy.deepcopy(doubleml_dict)
        test_dict['is_cluster_data'] = True
        test_dict['cluster_dict'] = {'cluster_ids': np.ones(shape=(n_obs, n_rep))}
        DoubleMLFramework(test_dict)

    test_dict = copy.deepcopy(doubleml_dict)
    framework_names = DoubleMLFramework(test_dict)

    msg = "treatment_names must be a list. Got 1 of type <class 'int'>."
    with pytest.raises(TypeError, match=msg):
        test_dict = copy.deepcopy(doubleml_dict)
        test_dict['treatment_names'] = 1
        DoubleMLFramework(test_dict)
    with pytest.raises(TypeError, match=msg):
        framework_names.treatment_names = 1

    msg = r"treatment_names must be a list of strings. At least one element is not a string: \['test', 1\]."
    with pytest.raises(TypeError, match=msg):
        test_dict = copy.deepcopy(doubleml_dict)
        test_dict['treatment_names'] = ['test', 1]
        DoubleMLFramework(test_dict)
    with pytest.raises(TypeError, match=msg):
        framework_names.treatment_names = ['test', 1]

    msg = "The length of treatment_names does not match the number of treatments. Got 2 treatments and 3 treatment names."
    with pytest.raises(ValueError, match=msg):
        test_dict = copy.deepcopy(doubleml_dict)
        test_dict['treatment_names'] = ['test', 'test2', 'test3']
        DoubleMLFramework(test_dict)
    with pytest.raises(ValueError, match=msg):
        framework_names.treatment_names = ['test', 'test2', 'test3']


def test_operation_exceptions():
    # addition
    msg = "Unsupported operand type: <class 'float'>"
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1 + 1.0
    with pytest.raises(TypeError, match=msg):
        _ = 1.0 + dml_framework_obj_1
    msg = 'The number of observations in DoubleMLFrameworks must be the same. Got 10 and 11.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs + 1, n_thetas, n_rep))
        psi_b_2 = np.random.normal(size=(n_obs + 1, n_thetas, n_rep))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = dml_framework_obj_1 + dml_framework_obj_2
    msg = 'The number of parameters theta in DoubleMLFrameworks must be the same. Got 2 and 3.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs, n_thetas + 1, n_rep))
        psi_b_2 = np.random.normal(size=(n_obs, n_thetas + 1, n_rep))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = dml_framework_obj_1 + dml_framework_obj_2
    msg = 'The number of replications in DoubleMLFrameworks must be the same. Got 5 and 6.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs, n_thetas, n_rep + 1))
        psi_b_2 = np.random.normal(size=(n_obs, n_thetas, n_rep + 1))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = dml_framework_obj_1 + dml_framework_obj_2

    # subtraction
    msg = "Unsupported operand type: <class 'float'>"
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1 - 1.0
    with pytest.raises(TypeError, match=msg):
        _ = 1.0 - dml_framework_obj_1
    msg = 'The number of observations in DoubleMLFrameworks must be the same. Got 10 and 11.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs + 1, n_thetas, n_rep))
        psi_b_2 = np.random.normal(size=(n_obs + 1, n_thetas, n_rep))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = dml_framework_obj_1 - dml_framework_obj_2
    msg = 'The number of parameters theta in DoubleMLFrameworks must be the same. Got 2 and 3.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs, n_thetas + 1, n_rep))
        psi_b_2 = np.random.normal(size=(n_obs, n_thetas + 1, n_rep))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = dml_framework_obj_1 - dml_framework_obj_2
    msg = 'The number of replications in DoubleMLFrameworks must be the same. Got 5 and 6.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs, n_thetas, n_rep + 1))
        psi_b_2 = np.random.normal(size=(n_obs, n_thetas, n_rep + 1))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = dml_framework_obj_1 - dml_framework_obj_2

    # multiplication
    msg = "Unsupported operand type: <class 'dict'>"
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1 * {}
    with pytest.raises(TypeError, match=msg):
        _ = {} * dml_framework_obj_1

    # concatenation
    msg = 'Need at least one object to concatenate.'
    with pytest.raises(TypeError, match=msg):
        concat([])
    msg = 'All objects must be of type DoubleMLFramework.'
    with pytest.raises(TypeError, match=msg):
        concat([dml_framework_obj_1, 1.0])
    msg = 'The number of observations in DoubleMLFrameworks must be the same. Got 10 and 11.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs + 1, n_thetas, n_rep))
        psi_b_2 = np.random.normal(size=(n_obs + 1, n_thetas, n_rep))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = concat([dml_framework_obj_1, dml_framework_obj_2])
    msg = 'The number of replications in DoubleMLFrameworks must be the same. Got 5 and 6.'
    with pytest.raises(ValueError, match=msg):
        psi_a_2 = np.ones(shape=(n_obs, n_thetas, n_rep + 1))
        psi_b_2 = np.random.normal(size=(n_obs, n_thetas, n_rep + 1))
        doubleml_dict_2 = generate_dml_dict(psi_a_2, psi_b_2)
        dml_framework_obj_2 = DoubleMLFramework(doubleml_dict_2)
        _ = concat([dml_framework_obj_1, dml_framework_obj_2])

    msg = 'concat not yet implemented with clustering.'
    with pytest.raises(NotImplementedError, match=msg):
        doubleml_dict_cluster = generate_dml_dict(psi_a_2, psi_b_2)
        doubleml_dict_cluster['is_cluster_data'] = True
        doubleml_dict_cluster['cluster_dict'] = {
            'smpls': np.ones(shape=(n_obs, n_rep)),
            'smpls_cluster': np.ones(shape=(n_obs, n_rep)),
            'cluster_vars': np.ones(shape=(n_obs, n_rep)),
            'n_folds_per_cluster': 2
        }
        dml_framework_obj_cluster = DoubleMLFramework(doubleml_dict_cluster)
        _ = concat([dml_framework_obj_cluster, dml_framework_obj_cluster])

    # cluster compatibility
    msg = "The cluster structure in DoubleMLFrameworks must be the same. Got False and True."
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_2 + dml_framework_obj_cluster


@pytest.mark.ci
def test_p_adjust_exceptions():
    msg = "The p_adjust method must be of str type. 1 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1.p_adjust(method=1)

    msg = r'Apply bootstrap\(\) before p_adjust\("rw"\)\.'
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1.p_adjust(method='rw')


@pytest.mark.ci
def test_sensitivity_exceptions():
    dml_framework_no_sensitivity = DoubleMLFramework(generate_dml_dict(psi_a, psi_b))
    msg = 'Sensitivity analysis is not implemented for this model.'
    with pytest.raises(NotImplementedError, match=msg):
        _ = dml_framework_no_sensitivity._calc_sensitivity_analysis(cf_y=0.1, cf_d=0.1, rho=1.0, level=0.95)

    # test cf_y
    msg = "cf_y must be of float type. 1 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1.sensitivity_analysis(cf_y=1)
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1._calc_sensitivity_analysis(cf_y=1, cf_d=0.03, rho=1.0, level=0.95)

    msg = r'cf_y must be in \[0,1\). 1.0 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1.sensitivity_analysis(cf_y=1.0)
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1._calc_sensitivity_analysis(cf_y=1.0, cf_d=0.03, rho=1.0, level=0.95)

    # test cf_d
    msg = "cf_d must be of float type. 1 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1.sensitivity_analysis(cf_y=0.1, cf_d=1)
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1._calc_sensitivity_analysis(cf_y=0.1, cf_d=1, rho=1.0, level=0.95)

    msg = r'cf_d must be in \[0,1\). 1.0 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1.sensitivity_analysis(cf_y=0.1, cf_d=1.0)
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1._calc_sensitivity_analysis(cf_y=0.1, cf_d=1.0, rho=1.0, level=0.95)

    # test rho
    msg = "rho must be of float type. 1 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1.sensitivity_analysis(cf_y=0.1, cf_d=0.15, rho=1)
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1._calc_sensitivity_analysis(cf_y=0.1, cf_d=0.15, rho=1, level=0.95)
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1._calc_robustness_value(rho=1, null_hypothesis=0.0, level=0.95, idx_treatment=0)

    msg = "rho must be of float type. 1 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1.sensitivity_analysis(cf_y=0.1, cf_d=0.15, rho="1")
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1._calc_sensitivity_analysis(cf_y=0.1, cf_d=0.15, rho="1", level=0.95)
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1._calc_robustness_value(rho="1", null_hypothesis=0.0, level=0.95, idx_treatment=0)

    msg = r'The absolute value of rho must be in \[0,1\]. 1.1 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1.sensitivity_analysis(cf_y=0.1, cf_d=0.15, rho=1.1)
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1._calc_sensitivity_analysis(cf_y=0.1, cf_d=0.15, rho=1.1, level=0.95)
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1._calc_robustness_value(rho=1.1, null_hypothesis=0.0, level=0.95, idx_treatment=0)

    # test level
    msg = "The confidence level must be of float type. 1 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1.sensitivity_analysis(cf_y=0.1, cf_d=0.15, rho=1.0, level=1)
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1._calc_sensitivity_analysis(cf_y=0.1, cf_d=0.15, rho=1.0, level=1)
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1._calc_robustness_value(rho=1.0, level=1, null_hypothesis=0.0, idx_treatment=0)

    msg = r'The confidence level must be in \(0,1\). 1.0 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1.sensitivity_analysis(cf_y=0.1, cf_d=0.15, rho=1.0, level=1.0)
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1._calc_sensitivity_analysis(cf_y=0.1, cf_d=0.15, rho=1.0, level=1.0)
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1._calc_robustness_value(rho=1.0, level=1.0, null_hypothesis=0.0, idx_treatment=0)

    msg = r'The confidence level must be in \(0,1\). 0.0 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1.sensitivity_analysis(cf_y=0.1, cf_d=0.15, rho=1.0, level=0.0)
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1._calc_sensitivity_analysis(cf_y=0.1, cf_d=0.15, rho=1.0, level=0.0)
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1._calc_robustness_value(rho=1.0, level=0.0, null_hypothesis=0.0, idx_treatment=0)

    # test null_hypothesis
    msg = "null_hypothesis has to be of type float or np.ndarry. 1 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1.sensitivity_analysis(null_hypothesis=1)
    msg = r"null_hypothesis is numpy.ndarray but does not have the required shape \(2,\). Array of shape \(3,\) was passed."
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1.sensitivity_analysis(null_hypothesis=np.array([1, 2, 3]))
    msg = "null_hypothesis must be of float type. 1 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1._calc_robustness_value(null_hypothesis=1, level=0.95, rho=1.0, idx_treatment=0)
    msg = r"null_hypothesis must be of float type. \[1\] of type <class 'numpy.ndarray'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1._calc_robustness_value(null_hypothesis=np.array([1]), level=0.95, rho=1.0, idx_treatment=0)

    sensitivity_dict = generate_dml_dict(psi_a, psi_b)
    sensitivity_dict['sensitivity_elements'] = {
        'sigma2': np.ones(shape=(1, n_thetas, n_rep)),
        'nu2': -1.0 * np.ones(shape=(1, n_thetas, n_rep)),
        'psi_sigma2': np.ones(shape=(n_obs, n_thetas, n_rep)),
        'psi_nu2': np.ones(shape=(n_obs, n_thetas, n_rep)),
        'riesz_rep': np.ones(shape=(n_obs, n_thetas, n_rep))
    }
    dml_framework_sensitivity = DoubleMLFramework(sensitivity_dict)

    # test idx_treatment
    dml_framework_obj_1.sensitivity_analysis()
    msg = "idx_treatment must be an integer. 0.0 of type <class 'float'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_1.sensitivity_plot(idx_treatment=0.0)

    msg = "idx_treatment must be larger or equal to 0. -1 was passed."
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1.sensitivity_plot(idx_treatment=-1)

    msg = "idx_treatment must be smaller or equal to 1. 2 was passed."
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_1.sensitivity_plot(idx_treatment=2)

    # test variances
    msg = (
        r'sensitivity_elements sigma2 and nu2 have to be positive\. '
        r'Got sigma2 \[\[\[1\. 1\. 1\. 1\. 1\.\]\n\s+\[1\. 1\. 1\. 1\. 1\.\]\]\] '
        r'and nu2 \[\[\[-1\. -1\. -1\. -1\. -1\.\]\n\s+\[-1\. -1\. -1\. -1\. -1\.\]\]\]\. '
        r'Most likely this is due to low quality learners \(especially propensity scores\)\.'
    )
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_sensitivity._calc_sensitivity_analysis(cf_y=0.03, cf_d=0.03, rho=1.0, level=0.95)
    with pytest.raises(ValueError, match=msg):
        dml_framework_sensitivity.sensitivity_analysis(cf_y=0.1, cf_d=0.15, rho=1.0, level=0.95)


@pytest.mark.ci
def test_framework_sensitivity_plot_input():
    dml_framework_obj_plot = DoubleMLFramework(doubleml_dict)

    msg = (r'Apply sensitivity_analysis\(\) to include senario in sensitivity_plot. ')
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot()

    dml_framework_obj_plot.sensitivity_analysis()
    msg = "null_hypothesis must be of float type. 1 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(null_hypothesis=1)

    msg = "include_scenario has to be boolean. True of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(include_scenario="True")

    msg = "benchmarks has to be either None or a dictionary. True of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(benchmarks="True")
    msg = r"benchmarks has to be a dictionary with keys cf_y, cf_d and name. Got dict_keys\(\['cf_y', 'cf_d'\]\)."
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(benchmarks={'cf_y': 0.1, 'cf_d': 0.15})
    msg = r"benchmarks has to be a dictionary with values of same length. Got \[1, 2, 2\]."
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(benchmarks={'cf_y': [0.1], 'cf_d': [0.15, 0.2],
                                                                'name': ['test', 'test2']})
    msg = "benchmarks cf_y must be of float type. 2 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(benchmarks={'cf_y': [0.1, 2], 'cf_d': [0.15, 0.2],
                                                                'name': ['test', 'test2']})
    msg = r'benchmarks cf_y must be in \[0,1\). 1.0 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(benchmarks={'cf_y': [0.1, 1.0], 'cf_d': [0.15, 0.2],
                                                                'name': ['test', 'test2']})
    msg = "benchmarks name must be of string type. 2 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(benchmarks={'cf_y': [0.1, 0.2], 'cf_d': [0.15, 0.2],
                                                                'name': [2, 2]})

    msg = "value must be a string. 2 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(value=2)
    msg = "Invalid value test. Valid values theta or ci."
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(value='test')

    msg = "fill has to be boolean. True of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(fill="True")

    msg = "grid_size must be an integer. 0.0 of type <class 'float'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(grid_size=0.0)
    msg = "grid_size must be larger or equal to 10. 9 was passed."
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(grid_size=9)

    msg = "grid_bounds must be of float type. 1 of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(grid_bounds=(0.15, 1))
    with pytest.raises(TypeError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(grid_bounds=(1, 0.15))
    msg = r'grid_bounds must be in \(0,1\). 1.0 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(grid_bounds=(1.0, 0.15))
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(grid_bounds=(0.15, 1.0))
    msg = r'grid_bounds must be in \(0,1\). 0.0 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(grid_bounds=(0.0, 0.15))
    with pytest.raises(ValueError, match=msg):
        _ = dml_framework_obj_plot.sensitivity_plot(grid_bounds=(0.15, 0.0))
