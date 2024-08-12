import pytest
import pandas as pd
import numpy as np

from doubleml import DoubleMLAPO, DoubleMLData
from doubleml.datasets import make_irm_data_discrete_treatments, make_iivm_data, make_irm_data

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

n = 100
data_apo = make_irm_data_discrete_treatments(n_obs=n)
df_apo = pd.DataFrame(np.column_stack((data_apo['y'], data_apo['d'], data_apo['x'])),
                      columns=['y', 'd'] + ['x' + str(i) for i in range(data_apo['x'].shape[1])])

dml_data = DoubleMLData(df_apo, 'y', 'd')

ml_g = Lasso()
ml_m = LogisticRegression()


@pytest.mark.ci
def test_apo_exception_data():
    msg = 'The data must be of DoubleMLData or DoubleMLClusterData type.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLAPO(pd.DataFrame(), ml_g, ml_m, treatment_level=0)

    msg = 'Only one treatment variable is allowed. Got 2 treatment variables.'
    with pytest.raises(ValueError, match=msg):
        dml_data_multiple = DoubleMLData(df_apo, 'y', ['d', 'x1'])
        _ = DoubleMLAPO(dml_data_multiple, ml_g, ml_m, treatment_level=0)

    dml_data_z = make_iivm_data()
    msg = r'Incompatible data. z have been set as instrumental variable\(s\).'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data_z, ml_g, ml_m, treatment_level=0)

    msg = 'The number of treated observations is less than 5. Number of treated observations: 0 for treatment level 1.1.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=1.1)

    msg = r'The proportion of observations with treatment level 42 is less than 5\%. Got 0.70\%.'
    # test warning
    with pytest.warns(UserWarning, match=msg):
        data_apo_warn = make_irm_data_discrete_treatments(n_obs=1000)
        data_apo_warn['d'][0:7] = 42
        df_apo_warn = pd.DataFrame(
            np.column_stack((data_apo_warn['y'], data_apo_warn['d'], data_apo_warn['x'])),
            columns=['y', 'd'] + ['x' + str(i) for i in range(data_apo_warn['x'].shape[1])]
        )
        dml_data_warn = DoubleMLData(df_apo_warn, 'y', 'd')

        _ = DoubleMLAPO(dml_data_warn, ml_g, ml_m, treatment_level=42)


@pytest.mark.ci
def test_apo_exception_learner():
    msg = (r'The ml_g learner LogisticRegression\(\) was identified as classifier but the outcome variable is not'
           ' binary with values 0 and 1.')
    with pytest.raises(ValueError, match=msg):
        ml_g_classifier = LogisticRegression()
        _ = DoubleMLAPO(dml_data, ml_g_classifier, ml_m, treatment_level=0)


@pytest.mark.ci
def test_apo_exception_scores():
    msg = 'Invalid score MAR. Valid score APO.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0, score='MAR')


@pytest.mark.ci
def test_apo_exception_trimming_rule():
    msg = 'Invalid trimming_rule discard. Valid trimming_rule truncate.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0, trimming_rule='discard')

    # check the trimming_threshold exceptions
    msg = "trimming_threshold has to be a float. Object of type <class 'str'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0,
                        trimming_rule='truncate', trimming_threshold="0.1")

    msg = 'Invalid trimming_threshold 0.6. trimming_threshold has to be between 0 and 0.5.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0,
                        trimming_rule='truncate', trimming_threshold=0.6)


@pytest.mark.ci
def test_apo_exception_ipw_normalization():
    msg = "Normalization indicator has to be boolean. Object of type <class 'int'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0, normalize_ipw=1)


@pytest.mark.ci
def test_apo_exception_weights():
    msg = "weights must be a numpy array or dictionary. weights of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0, weights=1)
    msg = r"weights must have keys \['weights', 'weights_bar'\]. keys dict_keys\(\['d'\]\) were passed."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0, weights={'d': [1, 2, 3]})

    # shape checks
    msg = rf"weights must have shape \({n},\). weights of shape \(1,\) was passed."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0, weights=np.ones(1))
    msg = rf"weights must have shape \({n},\). weights of shape \({n}, 2\) was passed."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0, weights=np.ones((n, 2)))

    msg = rf"weights must have shape \({n},\). weights of shape \(1,\) was passed."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0,
                        weights={'weights': np.ones(1), 'weights_bar': np.ones(1)})
    msg = rf"weights must have shape \({n},\). weights of shape \({n}, 2\) was passed."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0,
                        weights={'weights': np.ones((n, 2)), 'weights_bar': np.ones((n, 2))})
    msg = rf"weights_bar must have shape \({n}, 1\). weights_bar of shape \({n}, 2\) was passed."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0,
                        weights={'weights': np.ones(n), 'weights_bar': np.ones((n, 2))})

    # value checks
    msg = "All weights values must be greater or equal 0."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0,
                        weights=-1*np.ones(n,))
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0,
                        weights={'weights': -1*np.ones(n,), 'weights_bar': np.ones((n, 1))})
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0,
                        weights={'weights': np.ones(n,), 'weights_bar': -1*np.ones((n, 1))})

    msg = "At least one weight must be non-zero."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0,
                        weights=np.zeros((dml_data.d.shape[0], )))
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0,
                        weights={'weights': np.zeros((dml_data.d.shape[0], )),
                                 'weights_bar': np.ones((dml_data.d.shape[0], 1))})
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLAPO(dml_data, ml_g, ml_m, treatment_level=0,
                        weights={'weights': np.ones((dml_data.d.shape[0], )),
                                 'weights_bar': np.zeros((dml_data.d.shape[0], 1))})


@pytest.mark.ci
def test_apo_exception_capo_gapo():
    n = 20
    # collect data
    np.random.seed(42)
    obj_dml_data = make_irm_data(n_obs=n, dim_x=2)

    # First stage estimation
    ml_g = RandomForestRegressor(n_estimators=10)
    ml_m = RandomForestClassifier(n_estimators=10)

    dml_obj = DoubleMLAPO(obj_dml_data,
                          ml_m=ml_m,
                          ml_g=ml_g,
                          treatment_level=0)

    dml_obj.fit()
    # create a random basis
    random_basis = pd.DataFrame(np.random.normal(0, 1, size=(n, 5)))

    msg = "Invalid score APO_2. Valid score APO."
    with pytest.raises(ValueError, match=msg):
        dml_obj._score = 'APO_2'
        _ = dml_obj.capo(random_basis)
    # reset the score
    dml_obj._score = 'APO'

    msg = "Only implemented for one repetition. Number of repetitions is 2."
    with pytest.raises(NotImplementedError, match=msg):
        dml_obj._n_rep = 2
        dml_obj.capo(random_basis)
    # reset the number of repetitions
    dml_obj._n_rep = 1

    msg = "Groups must be of DataFrame type. Groups of type <class 'int'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = dml_obj.gapo(1)

    groups_1 = pd.DataFrame(
        np.column_stack([obj_dml_data.data['X1'] > 0.2, np.ones_like(obj_dml_data.data['X1'])]),
        columns=['Group 1', 'Group 2']
    )
    msg = (r'Columns of groups must be of bool type or int type \(dummy coded\). Alternatively,'
           ' groups should only contain one column.')
    with pytest.raises(TypeError, match=msg):
        _ = dml_obj.gapo(groups_1)
