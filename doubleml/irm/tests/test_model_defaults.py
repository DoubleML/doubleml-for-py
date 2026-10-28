import numpy as np
import pytest
from sklearn.linear_model import Lasso, LogisticRegression

import doubleml as dml
from doubleml.irm.datasets import make_irm_data
from doubleml.utils._check_defaults import _check_basic_defaults_after_fit, _check_basic_defaults_before_fit, _fit_bootstrap

np.random.seed(3141)
dml_data_irm = make_irm_data(n_obs=500)


@pytest.mark.ci
def test_irm_defaults():
    """Check the default legacy DoubleMLIRM configuration before and after fitting."""
    dml_irm = dml.DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression())

    _check_basic_defaults_before_fit(dml_irm)
    assert dml_irm.draw_sample_splitting

    _fit_bootstrap(dml_irm)
    _check_basic_defaults_after_fit(dml_irm)

    assert dml_irm.score == "ATE"
    assert isinstance(dml_irm.ps_processor_config, dml.utils.PSProcessorConfig)
    assert isinstance(dml_irm.ps_processor, dml.utils.PSProcessor)
    assert not dml_irm.normalize_ipw
    assert set(dml_irm.weights.keys()) == {"weights"}
    assert np.array_equal(dml_irm.weights["weights"], np.ones((dml_irm._dml_data.n_obs,)))


@pytest.mark.ci
def test_policytree_defaults():
    """Check the default policy-tree hyperparameters produced by DoubleMLIRM."""
    dml_irm = dml.DoubleMLIRM(dml_data_irm, Lasso(), LogisticRegression())
    dml_irm.fit()
    policy_tree = dml_irm.policy_tree(features=dml_data_irm.data.drop(columns=["y", "d"]))

    assert policy_tree.policy_tree.max_depth == 2
    assert policy_tree.policy_tree.min_samples_leaf == 8
    assert policy_tree.policy_tree.ccp_alpha == 0.01
