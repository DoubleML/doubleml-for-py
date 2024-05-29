import pytest
import pandas as pd
import numpy as np

from doubleml import DoubleMLSSM
from doubleml.datasets import make_ssm_data
from doubleml.double_ml_data import DoubleMLBaseData

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.base import BaseEstimator

np.random.seed(3141)
n = 100
dml_data_mar = make_ssm_data(n_obs=n, mar=True)
dml_data_nonignorable = make_ssm_data(n_obs=n, mar=False)
ml_g = Lasso()
ml_pi = LogisticRegression()
ml_m = LogisticRegression()
dml_ssm_mar = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, ml_m)
dml_ssm_nonignorable = DoubleMLSSM(dml_data_nonignorable, ml_g, ml_pi, ml_m)


class DummyDataClass(DoubleMLBaseData):
    def __init__(self,
                 data):
        DoubleMLBaseData.__init__(self, data)

    @property
    def n_coefs(self):
        return 1


@pytest.mark.ci
def test_ssm_exception_data():
    msg = 'The data must be of DoubleMLData or DoubleMLClusterData type.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLSSM(pd.DataFrame(), ml_g, ml_pi, ml_m)

    msg = 'The data must be of DoubleMLData type.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLSSM(DummyDataClass(pd.DataFrame(np.zeros((100, 10)))), ml_g, ml_pi, ml_m)

    # Nonignorable nonresponse without instrument
    msg = ('Sample selection by nonignorable nonresponse was set but instrumental variable \
                             is None. To estimate treatment effect under nonignorable nonresponse, \
                             specify an instrument for the selection variable.')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, Lasso(), LogisticRegression(), LogisticRegression(), score='nonignorable')


@pytest.mark.ci
def test_ssm_exception_scores():
    # MAR
    msg = 'Invalid score MAR. Valid score missing-at-random or nonignorable.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, ml_m, score='MAR')
    msg = 'score should be either a string or a callable. 0 was passed.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, ml_m, score=0)


@pytest.mark.ci
def test_ssm_exception_trimming_rule():
    msg = 'Invalid trimming_rule discard. Valid trimming_rule truncate.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, ml_m, trimming_rule='discard')

    # check the trimming_threshold exceptions
    msg = "trimming_threshold has to be a float. Object of type <class 'str'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, ml_m,
                        trimming_rule='truncate', trimming_threshold="0.1")

    msg = 'Invalid trimming_threshold 0.6. trimming_threshold has to be between 0 and 0.5.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, ml_m,
                        trimming_rule='truncate', trimming_threshold=0.6)


@pytest.mark.ci
def test_ssm_exception_ipw_normalization():
    msg = "Normalization indicator has to be boolean. Object of type <class 'int'> passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, ml_m, normalize_ipw=1)


@pytest.mark.ci
def test_ssm_exception_resampling():
    msg = "The number of folds must be of int type. 1.5 of type <class 'float'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, ml_m, n_folds=1.5)
    msg = ('The number of repetitions for the sample splitting must be of int type. '
           "1.5 of type <class 'float'> was passed.")
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, ml_m, n_rep=1.5)
    msg = 'The number of folds must be positive. 0 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, ml_m, n_folds=0)
    msg = 'The number of repetitions for the sample splitting must be positive. 0 was passed.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, ml_m, n_rep=0)
    msg = 'draw_sample_splitting must be True or False. Got true.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, ml_m, draw_sample_splitting='true')


@pytest.mark.ci
def test_ssm_exception_get_params():
    msg = 'Invalid nuisance learner ml_r. Valid nuisance learner ml_g_d0 or ml_g_d1 or ml_pi or ml_m.'
    with pytest.raises(ValueError, match=msg):
        dml_ssm_mar.get_params('ml_r')


@pytest.mark.ci
def test_ssm_exception_smpls():
    msg = ('Sample splitting not specified. '
           r'Either draw samples via .draw_sample splitting\(\) or set external samples via .set_sample_splitting\(\).')
    dml_plr_no_smpls = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, ml_m, draw_sample_splitting=False)
    with pytest.raises(ValueError, match=msg):
        _ = dml_plr_no_smpls.smpls


@pytest.mark.ci
def test_ssm_exception_fit():
    msg = "The number of CPUs used to fit the learners must be of int type. 5 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_ssm_mar.fit(n_jobs_cv='5')
    msg = 'store_predictions must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_ssm_mar.fit(store_predictions=1)
    msg = 'store_models must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_ssm_mar.fit(store_models=1)


@pytest.mark.ci
def test_ssm_exception_bootstrap():
    dml_ssm_boot = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, ml_m)
    msg = r'Apply fit\(\) before bootstrap\(\).'
    with pytest.raises(ValueError, match=msg):
        dml_ssm_boot.bootstrap()

    dml_ssm_boot.fit()
    msg = 'Method must be "Bayes", "normal" or "wild". Got Gaussian.'
    with pytest.raises(ValueError, match=msg):
        dml_ssm_boot.bootstrap(method='Gaussian')
    msg = "The number of bootstrap replications must be of int type. 500 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_ssm_boot.bootstrap(n_rep_boot='500')
    msg = 'The number of bootstrap replications must be positive. 0 was passed.'
    with pytest.raises(ValueError, match=msg):
        dml_ssm_boot.bootstrap(n_rep_boot=0)


@pytest.mark.ci
def test_ssm_exception_confint():
    dml_ssm_confint = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, ml_m)
    msg = r'Apply fit\(\) before confint\(\).'
    with pytest.raises(ValueError, match=msg):
        dml_ssm_confint.confint()
    dml_ssm_confint.fit()

    msg = 'joint must be True or False. Got 1.'
    with pytest.raises(TypeError, match=msg):
        dml_ssm_confint.confint(joint=1)
    msg = "The confidence level must be of float type. 5% of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_ssm_confint.confint(level='5%')
    msg = r'The confidence level must be in \(0,1\). 0.0 was passed.'
    with pytest.raises(ValueError, match=msg):
        dml_ssm_confint.confint(level=0.)

    msg = r'Apply bootstrap\(\) before confint\(joint=True\).'
    with pytest.raises(ValueError, match=msg):
        dml_ssm_confint.confint(joint=True)
    dml_ssm_confint.bootstrap()
    df_ssm_ci = dml_ssm_confint.confint(joint=True)
    assert isinstance(df_ssm_ci, pd.DataFrame)


@pytest.mark.ci
def test_ssm_exception_set_ml_nuisance_params():
    msg = 'Invalid nuisance learner g. Valid nuisance learner ml_g_d0 or ml_g_d1 or ml_pi or ml_m.'
    with pytest.raises(ValueError, match=msg):
        dml_ssm_mar.set_ml_nuisance_params('g', 'd', {'alpha': 0.1})
    msg = 'Invalid treatment variable y. Valid treatment variable d.'
    with pytest.raises(ValueError, match=msg):
        dml_ssm_mar.set_ml_nuisance_params('ml_g_d0', 'y', {'alpha': 0.1})


class _DummyNoSetParams:
    def fit(self):
        pass


class _DummyNoGetParams(_DummyNoSetParams):
    def set_params(self):
        pass


class _DummyNoClassifier(_DummyNoGetParams):
    def get_params(self):
        pass

    def predict_proba(self):
        pass


@pytest.mark.ci
def test_ssm_exception_learner():
    err_msg_prefix = 'Invalid learner provided for ml_g: '

    msg = err_msg_prefix + 'provide an instance of a learner instead of a class.'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, Lasso, ml_pi, ml_m)
    msg = err_msg_prefix + r'BaseEstimator\(\) has no method .fit\(\).'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, BaseEstimator(), ml_pi, ml_m)
    msg = r'has no method .set_params\(\).'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, _DummyNoSetParams(), ml_pi, ml_m)
    msg = r'has no method .get_params\(\).'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, _DummyNoGetParams(), ml_pi, ml_m)

    # allow classifiers for ml_g, but only for binary outcome
    msg = (r'The ml_g learner LogisticRegression\(\) was identified as classifier '
           'but the outcome is not binary with values 0 and 1.')
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, LogisticRegression(), ml_pi, ml_m)

    # construct a classifier which is not identifiable as classifier via is_classifier by sklearn
    # it then predicts labels and therefore an exception will be thrown
    log_reg = LogisticRegression()
    log_reg._estimator_type = None
    msg = (r'Learner provided for ml_m is probably invalid: LogisticRegression\(\) is \(probably\) no classifier.')
    with pytest.warns(UserWarning, match=msg):
        _ = DoubleMLSSM(dml_data_mar, ml_g, ml_pi, log_reg)


@pytest.mark.ci
def test_ssm_exception_and_warning_learner():
    # msg = err_msg_prefix + r'_DummyNoClassifier\(\) has no method .predict\(\).'
    with pytest.raises(TypeError):
        _ = DoubleMLSSM(dml_data_mar, _DummyNoClassifier(), ml_pi, ml_m)
    msg = 'Invalid learner provided for ml_pi: ' + r'Lasso\(\) has no method .predict_proba\(\).'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, Lasso(), Lasso(), ml_m)
    msg = 'Invalid learner provided for ml_m: ' + r'Lasso\(\) has no method .predict_proba\(\).'
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, Lasso(), ml_pi, Lasso())


class LassoWithNanPred(Lasso):
    def predict(self, X):
        preds = super().predict(X)
        n_obs = len(preds)
        preds[np.random.randint(0, n_obs, 1)] = np.nan
        return preds


class LassoWithInfPred(Lasso):
    def predict(self, X):
        preds = super().predict(X)
        n_obs = len(preds)
        preds[np.random.randint(0, n_obs, 1)] = np.inf
        return preds


@pytest.mark.ci
def test_ssm_nan_prediction():
    msg = r'Predictions from learner LassoWithNanPred\(\) for ml_g_d1 are not finite.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, LassoWithNanPred(), ml_pi, ml_m).fit()
    msg = r'Predictions from learner LassoWithInfPred\(\) for ml_g_d1 are not finite.'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLSSM(dml_data_mar, LassoWithInfPred(), ml_pi, ml_m).fit()


@pytest.mark.ci
def test_double_ml_exception_evaluate_learner():
    dml_ssm_obj = DoubleMLSSM(dml_data_mar,
                              ml_g=Lasso(),
                              ml_pi=LogisticRegression(),
                              ml_m=LogisticRegression(),
                              trimming_threshold=0.05,
                              n_folds=5,
                              score='missing-at-random')

    msg = r'Apply fit\(\) before evaluate_learners\(\).'
    with pytest.raises(ValueError, match=msg):
        dml_ssm_obj.evaluate_learners()

    dml_ssm_obj.fit()

    msg = "metric should be a callable. 'mse' was passed."
    with pytest.raises(TypeError, match=msg):
        dml_ssm_obj.evaluate_learners(metric="mse")

    msg = (r"The learners have to be a subset of \['ml_g_d0', 'ml_g_d1', 'ml_pi', 'ml_m'\]. "
           r"Learners \['ml_mu', 'ml_p'\] provided.")
    with pytest.raises(ValueError, match=msg):
        dml_ssm_obj.evaluate_learners(learners=['ml_mu', 'ml_p'])

    msg = 'Evaluation from learner ml_g_d0 is not finite.'

    def eval_fct(y_pred, y_true):
        return np.nan
    with pytest.raises(ValueError, match=msg):
        dml_ssm_obj.evaluate_learners(metric=eval_fct)
