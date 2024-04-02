import numpy as np
import pandas as pd
import pytest
import copy

import doubleml as dml

from ._utils_pt_manual import fit_policytree
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError


@pytest.fixture(scope='module',
                params=[1, 2, 3])
def depth(request):
    return request.param


@pytest.fixture(scope='module')
def dml_policytree_fixture(depth):
    n = 50
    np.random.seed(42)
    random_x_var = pd.DataFrame(np.random.normal(0, 1, size=(n, 3)))
    random_signal = np.random.normal(0, 1, size=(n, ))

    policy_tree = dml.DoubleMLPolicyTree(random_signal, random_x_var, depth)

    policy_tree_obj = copy.copy(policy_tree)
    np.random.seed(42)
    policy_tree.fit()
    np.random.seed(42)
    policy_tree_manual = fit_policytree(random_signal, random_x_var, depth)

    res_dict = {'tree': policy_tree.policy_tree.tree_,
                'tree_manual': policy_tree_manual.tree_,
                'features': policy_tree.features,
                'signal': policy_tree.orth_signal,
                'policytree_model': policy_tree,
                'unfitted_policytree_model': policy_tree_obj}

    return res_dict


@pytest.mark.ci
def test_dml_policytree_treshold(dml_policytree_fixture):
    assert np.allclose(dml_policytree_fixture['tree'].threshold,
                       dml_policytree_fixture['tree_manual'].threshold,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_policytree_children(dml_policytree_fixture):
    assert np.allclose(dml_policytree_fixture['tree'].children_left,
                       dml_policytree_fixture['tree_manual'].children_left,
                       rtol=1e-9, atol=1e-4)
    assert np.allclose(dml_policytree_fixture['tree'].children_right,
                       dml_policytree_fixture['tree_manual'].children_right,
                       rtol=1e-9, atol=1e-4)


@pytest.mark.ci
def test_dml_policytree_return_types(dml_policytree_fixture):
    assert isinstance(dml_policytree_fixture['policytree_model'].__str__(), str)
    assert isinstance(dml_policytree_fixture['policytree_model'].summary, pd.DataFrame)
    assert isinstance(dml_policytree_fixture['policytree_model'].policy_tree, DecisionTreeClassifier)


@pytest.mark.ci
def test_doubleml_exception_policytree():
    random_features = pd.DataFrame(np.random.normal(0, 1, size=(2, 3)), columns=['a', 'b', 'c'])
    signal = np.array([1, 2])

    msg = "The signal must be of np.ndarray type. Signal of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml.DoubleMLPolicyTree(orth_signal=1, features=random_features)
    msg = 'The signal must be of one dimensional. Signal of dimensions 2 was passed.'
    with pytest.raises(ValueError, match=msg):
        dml.DoubleMLPolicyTree(orth_signal=np.array([[1], [2]]), features=random_features)
    msg = "The features must be of DataFrame type. Features of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml.DoubleMLPolicyTree(orth_signal=signal, features=1)
    msg = 'Invalid pd.DataFrame: Contains duplicate column names.'
    with pytest.raises(ValueError, match=msg):
        dml.DoubleMLPolicyTree(orth_signal=signal, features=pd.DataFrame(np.array([[1, 2], [4, 5]]),
                                                                         columns=['a_1', 'a_1']))

    dml_policytree_predict = dml.DoubleMLPolicyTree(orth_signal=signal, features=random_features)
    msg = 'Policy Tree not yet fitted. Call fit before predict.'
    with pytest.raises(NotFittedError, match=msg):
        dml_policytree_predict.predict(random_features)

    dml_policytree_predict.fit()
    msg = "The features must be of DataFrame type. Features of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_policytree_predict.predict(features=1)
    msg = (r'The features must have the keys Index\(\[\'a\', \'b\', \'c\'\], dtype\=\'object\'\). '
           r'Features with keys Index\(\[\'d\'\], dtype=\'object\'\) were passed.')
    with pytest.raises(KeyError, match=msg):
        dml_policytree_predict.predict(features=pd.DataFrame({"d": [3, 4]}))

    dml_policytree_plot = dml.DoubleMLPolicyTree(orth_signal=signal, features=random_features)
    msg = 'Policy Tree not yet fitted. Call fit before plot_tree.'
    with pytest.raises(NotFittedError, match=msg):
        dml_policytree_plot.plot_tree()
