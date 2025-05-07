import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml

np.random.seed(3141)

@pytest.fixture(scope="module")
def true_ATE():
    return 0.5

@pytest.fixture(scope="module")
def instrument_size():
    return 0.005

@pytest.fixture(scope="module")
def n_samples():
    return 1000

@pytest.fixture(scope="module")
def n_simulations():
    return 100

@pytest.fixture(scope="module")
def weakiv_data(n_samples, instrument_size, true_ATE):
    # Generate data
    u = np.random.normal(0, 2, size=n_samples)
    X = np.random.normal(0, 1, size=n_samples)
    Z = np.random.binomial(1, 0.5, size=n_samples)
    A = instrument_size * Z + u # Continuous treatment A
    A = np.array(A > 0, dtype=int)
    Y = true_ATE * A + np.sign(u) # Outcome Y
    return dml.DoubleMLData.from_arrays(x=X, y=Y, d=A, z=Z)


@pytest.fixture(scope="module")
def iivm_obj(weakiv_data):
    # Set machine learning methods for m, r & g
    learner_g = LinearRegression()
    classifier_m = LogisticRegression()
    classifier_r = RandomForestClassifier(n_estimators=20, max_depth=5)

    # Create DoubleMLIIVM object
    obj_dml_data = weakiv_data
    dml_iivm_obj = dml.DoubleMLIIVM(obj_dml_data, learner_g, classifier_m, classifier_r)
    return dml_iivm_obj


def test_coverage(iivm_obj, true_ATE, n_simulations):
    coverage = []
    for _ in range(n_simulations):
        # Fit the model
        iivm_obj.fit()

        # Get the confidence set
        conf_set = iivm_obj.uniform_confset()

        # Check if the true ATE is in the confidence set
        ate_in_confset = any(x[0] < true_ATE < x[1] for x in conf_set)
        coverage.append(ate_in_confset)
    
    # Calculate the coverage rate
    coverage_rate = np.mean(coverage)
    assert coverage_rate >= 0.9, f"Coverage rate {coverage_rate} is below 0.9"

