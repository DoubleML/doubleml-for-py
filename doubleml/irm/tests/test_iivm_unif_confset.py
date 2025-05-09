import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml


def generate_weak_iv_data(n_samples, instrument_size, true_ATE):
    u = np.random.normal(0, 2, size=n_samples)
    X = np.random.normal(0, 1, size=n_samples)
    Z = np.random.binomial(1, 0.5, size=n_samples)
    A = instrument_size * Z + u
    A = np.array(A > 0, dtype=int)
    Y = true_ATE * A + np.sign(u)
    dml_data = dml.DoubleMLData.from_arrays(x=X, y=Y, d=A, z=Z)
    return dml_data


@pytest.mark.ci
def test_coverage_robust_confset():
    # Test parameters
    true_ATE = 0.5
    instrument_size = 0.005
    n_samples = 1000
    n_simulations = 100

    np.random.seed(3141)
    coverage = []
    for _ in range(n_simulations):
        data = generate_weak_iv_data(n_samples, instrument_size, true_ATE)

        # Set machine learning methods
        learner_g = LinearRegression()
        classifier_m = LogisticRegression()
        classifier_r = RandomForestClassifier(n_estimators=20, max_depth=5)

        # Create and fit new model
        dml_iivm_obj = dml.DoubleMLIIVM(data, learner_g, classifier_m, classifier_r)
        dml_iivm_obj.fit()

        # Get confidence set
        conf_set = dml_iivm_obj.robust_confset()

        # check if conf_set is a list of tuples
        assert isinstance(conf_set, list)
        assert all(isinstance(x, tuple) and len(x) == 2 for x in conf_set)

        # Check if true ATE is in confidence set
        ate_in_confset = any(x[0] < true_ATE < x[1] for x in conf_set)
        coverage.append(ate_in_confset)

    # Calculate coverage rate
    coverage_rate = np.mean(coverage)
    assert coverage_rate >= 0.9, f"Coverage rate {coverage_rate} is below 0.9"


@pytest.mark.ci
def test_exceptions_robust_confset():
    # Test parameters
    true_ATE = 0.5
    instrument_size = 0.005
    n_samples = 1000

    np.random.seed(3141)
    data = generate_weak_iv_data(n_samples, instrument_size, true_ATE)

    # create new model
    learner_g = LinearRegression()
    classifier_m = LogisticRegression()
    classifier_r = RandomForestClassifier(n_estimators=20, max_depth=5)
    dml_iivm_obj = dml.DoubleMLIIVM(data, learner_g, classifier_m, classifier_r)

    # Check if the robust_confset method raises an exception when called before fitting
    msg = r"Apply fit\(\) before robust_confset\(\)."
    with pytest.raises(ValueError, match=msg):
        dml_iivm_obj.robust_confset()

    # Check if str representation of the object is working
    str_repr = str(dml_iivm_obj)
    assert isinstance(str_repr, str)
    assert "Robust" not in str_repr

    # Fit the model
    dml_iivm_obj.fit()

    # Check invalid inputs
    msg = "The confidence level must be of float type. 0.95 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_iivm_obj.robust_confset(level="0.95")
    msg = r"The confidence level must be in \(0,1\). 1.5 was passed."
    with pytest.raises(ValueError, match=msg):
        dml_iivm_obj.robust_confset(level=1.5)

    # Check if str representation of the object is working
    str_repr = str(dml_iivm_obj)
    assert isinstance(str_repr, str)
    assert "Robust Confidence Set" in str_repr
