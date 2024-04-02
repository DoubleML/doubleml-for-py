import pytest
import numpy as np
import pandas as pd

from doubleml import DoubleMLQTE, DoubleMLData
from doubleml.datasets import make_irm_data
from doubleml.double_ml_data import DoubleMLBaseData

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
n = 100
dml_data_irm = make_irm_data(n_obs=n)
ml_m = Lasso()
ml_g = RandomForestClassifier()


class DummyDataClass(DoubleMLBaseData):
    def __init__(self,
                 data):
        DoubleMLBaseData.__init__(self, data)

    @property
    def n_coefs(self):
        return 1


@pytest.mark.ci
def test_exception_data():
    msg = 'The data must be of DoubleMLData type.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLQTE(DummyDataClass(pd.DataFrame(np.zeros((100, 10)))), ml_g, ml_m)

    msg = ('Incompatible data. To fit an PQ model with DML exactly one binary variable with values 0 and 1 '
           'needs to be specified as treatment variable.')
    df_irm = dml_data_irm.data.copy()
    df_irm['d'] = df_irm['d'] * 2
    with pytest.raises(ValueError, match=msg):
        # non-binary D for QTE
        _ = DoubleMLQTE(DoubleMLData(df_irm, 'y', 'd'),
                        LogisticRegression(), LogisticRegression())
    df_irm = dml_data_irm.data.copy()
    with pytest.raises(ValueError, match=msg):
        # multiple D for QTE
        _ = DoubleMLQTE(DoubleMLData(df_irm, 'y', ['d', 'X1']),
                        LogisticRegression(), LogisticRegression())


@pytest.mark.ci
def test_exception_score():
    # QTE
    msg = 'Invalid score IV. Valid score PQ or LPQ or CVaR.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLQTE(dml_data_irm, LogisticRegression(), LogisticRegression(), score='IV')
    msg = 'score should be a string. 2 was passed.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLQTE(dml_data_irm, LogisticRegression(), LogisticRegression(), score=2)


@pytest.mark.ci
def test_exception_trimming_rule():
    msg = 'Invalid trimming_rule discard. Valid trimming_rule truncate.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLQTE(dml_data_irm, LogisticRegression(), LogisticRegression(), trimming_rule='discard')

    msg = "trimming_threshold has to be a float. Object of type <class 'str'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLQTE(dml_data_irm, LogisticRegression(), LogisticRegression(),
                        trimming_rule='truncate', trimming_threshold="0.1")

    msg = 'Invalid trimming_threshold 0.6. trimming_threshold has to be between 0 and 0.5.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLQTE(dml_data_irm, LogisticRegression(), LogisticRegression(),
                        trimming_rule='truncate', trimming_threshold=0.6)


@pytest.mark.ci
def test_exception_quantiles():
    msg = r'Quantiles have be between 0 or 1. Quantiles \[0.2 2. \] passed.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLQTE(dml_data_irm, ml_g, ml_m, quantiles=[0.2, 2])


@pytest.mark.ci
def test_exception_ipw_normalization():
    msg = "Normalization indicator has to be boolean. Object of type <class 'int'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLQTE(dml_data_irm, ml_g, ml_m, normalize_ipw=1)


@pytest.mark.ci
def test_exception_bootstrap():
    dml_qte_boot = DoubleMLQTE(dml_data_irm, RandomForestClassifier(), RandomForestClassifier())
    msg = r'Apply fit\(\) before bootstrap\(\).'
    with pytest.raises(ValueError, match=msg):
        dml_qte_boot.bootstrap()

    dml_qte_boot.fit()
    msg = 'Method must be "Bayes", "normal" or "wild". Got Gaussian.'
    with pytest.raises(ValueError, match=msg):
        dml_qte_boot.bootstrap(method='Gaussian')
    msg = "The number of bootstrap replications must be of int type. 500 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_qte_boot.bootstrap(n_rep_boot='500')
    msg = 'The number of bootstrap replications must be positive. 0 was passed.'
    with pytest.raises(ValueError, match=msg):
        dml_qte_boot.bootstrap(n_rep_boot=0)


@pytest.mark.ci
def test_doubleml_exception_confint():
    dml_qte_confint = DoubleMLQTE(dml_data_irm, RandomForestClassifier(), RandomForestClassifier())
    dml_qte_confint.fit()

    msg = 'joint must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_qte_confint.confint(joint=1)
    msg = "The confidence level must be of float type. 5% of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_qte_confint.confint(level='5%')
    msg = r'The confidence level must be in \(0,1\). 0.0 was passed.'
    with pytest.raises(ValueError, match=msg):
        dml_qte_confint.confint(level=0.)

    dml_qte_confint_not_fitted = DoubleMLQTE(dml_data_irm, RandomForestClassifier(), RandomForestClassifier())
    msg = r'Apply fit\(\) before confint\(\).'
    with pytest.raises(ValueError, match=msg):
        dml_qte_confint_not_fitted.confint()
    msg = r'Apply bootstrap\(\) before confint\(joint=True\).'
    with pytest.raises(ValueError, match=msg):
        dml_qte_confint.confint(joint=True)
    dml_qte_confint.bootstrap()
    df_qte_ci = dml_qte_confint.confint(joint=True)
    assert isinstance(df_qte_ci, pd.DataFrame)


@pytest.mark.ci
def test_doubleml_exception_p_adjust():
    dml_qte_p_adjust = DoubleMLQTE(dml_data_irm, RandomForestClassifier(), RandomForestClassifier())

    msg = r'Apply fit\(\) before p_adjust\(\).'
    with pytest.raises(ValueError, match=msg):
        dml_qte_p_adjust.p_adjust()
    dml_qte_p_adjust.fit()
    msg = r'Apply bootstrap\(\) before p_adjust\("romano-wolf"\).'
    with pytest.raises(ValueError, match=msg):
        dml_qte_p_adjust.p_adjust(method='romano-wolf')
    dml_qte_p_adjust.bootstrap()
    p_val = dml_qte_p_adjust.p_adjust(method='romano-wolf')
    assert isinstance(p_val, pd.DataFrame)

    msg = "The p_adjust method must be of str type. 0.05 of type <class 'float'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_qte_p_adjust.p_adjust(method=0.05)
