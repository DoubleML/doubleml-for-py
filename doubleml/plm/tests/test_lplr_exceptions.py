import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.semi_supervised import LabelSpreading

from doubleml import DoubleMLLPLR
from doubleml.plm.datasets import make_lplr_LZZ2020

np.random.seed(3141)
n = 100
# create test data and basic learners
dml_data = make_lplr_LZZ2020(alpha=0.5, n_obs=n, dim_x=20)
dml_data_binary = make_lplr_LZZ2020(alpha=0.5, n_obs=n, treatment="binary", dim_x=20)
ml_M = RandomForestClassifier(max_depth=2, n_estimators=10)
ml_t = RandomForestRegressor(max_depth=2, n_estimators=10)
ml_m = RandomForestRegressor(max_depth=2, n_estimators=10)
dml_lplr = DoubleMLLPLR(dml_data, ml_M, ml_t, ml_m)
dml_lplr_instrument = DoubleMLLPLR(dml_data, ml_M, ml_t, ml_m, score="instrument")


@pytest.mark.ci
def test_lplr_exception_data():
    msg = r"The data must be of DoubleMLData.* type\.[\s\S]* of type " r"<class 'pandas\.core\.frame\.DataFrame'> was passed\."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPLR(pd.DataFrame(), ml_M, ml_t, ml_m)

    dml_data_nb = make_lplr_LZZ2020(alpha=0.5, n_obs=50, dim_x=20)
    dml_data_nb.data[dml_data_nb.y_col] = dml_data_nb.data[dml_data_nb.y_col] + 1
    dml_data_nb._set_y_z()
    with pytest.raises(TypeError, match="The outcome variable y must be binary with values 0 and 1."):
        _ = DoubleMLLPLR(dml_data_nb, ml_M, ml_t, ml_m)


@pytest.mark.ci
def test_lplr_exception_scores():
    # LPLR valid scores are 'nuisance_space' and 'instrument'
    msg = "Invalid score MAR"
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, ml_t, ml_m, score="MAR")
    msg = "score should be a string. 0 was passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, ml_t, ml_m, score=0)


@pytest.mark.ci
def test_lplr_exception_resampling():
    msg = "The number of folds must be of int type. 1.5 of type <class 'float'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, ml_t, ml_m, n_folds=1.5)

    msg = "The number of repetitions for the sample splitting must be of int type. 1.5 of type <class 'float'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, ml_t, ml_m, n_rep=1.5)

    msg = "The number of folds must be positive. 0 was passed."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, ml_t, ml_m, n_folds=0)

    msg = "The number of repetitions for the sample splitting must be positive. 0 was passed."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, ml_t, ml_m, n_rep=0)

    msg = "draw_sample_splitting must be True or False. Got true."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, ml_t, ml_m, draw_sample_splitting="true")


@pytest.mark.ci
def test_lplr_exception_get_params():
    msg = r"Invalid nuisance learner ml_x. Valid nuisance learner ml_m or ml_a or ml_t or ml_M.*"
    with pytest.raises(ValueError, match=msg):
        dml_lplr.get_params("ml_x")


@pytest.mark.ci
def test_lplr_exception_smpls():
    msg = (
        "Sample splitting not specified. "
        r"Either draw samples via .draw_sample splitting\(\) or set external samples via .set_sample_splitting\(\)."
    )
    dml_plr_no_smpls = DoubleMLLPLR(dml_data, ml_M, ml_t, ml_m, draw_sample_splitting=False)
    with pytest.raises(ValueError, match=msg):
        _ = dml_plr_no_smpls.smpls


@pytest.mark.ci
def test_lplr_exception_fit():
    msg = "The number of CPUs used to fit the learners must be of int type. 5 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_lplr.fit(n_jobs_cv="5")
    msg = "store_predictions must be True or False. Got 1."
    with pytest.raises(TypeError, match=msg):
        dml_lplr.fit(store_predictions=1)
    msg = "store_models must be True or False. Got 1."
    with pytest.raises(TypeError, match=msg):
        dml_lplr.fit(store_models=1)


@pytest.mark.ci
def test_lplr_exception_bootstrap():
    dml_lplr_boot = DoubleMLLPLR(dml_data, ml_M, ml_t, ml_m)
    msg = r"Apply fit\(\) before bootstrap\(\)."
    with pytest.raises(ValueError, match=msg):
        dml_lplr_boot.bootstrap()

    dml_lplr_boot.fit()
    msg = 'Method must be "Bayes", "normal" or "wild". Got Gaussian.'
    with pytest.raises(ValueError, match=msg):
        dml_lplr_boot.bootstrap(method="Gaussian")
    msg = "The number of bootstrap replications must be of int type. 500 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_lplr_boot.bootstrap(n_rep_boot="500")
    msg = "The number of bootstrap replications must be positive. 0 was passed."
    with pytest.raises(ValueError, match=msg):
        dml_lplr_boot.bootstrap(n_rep_boot=0)


@pytest.mark.ci
def test_lplr_exception_confint():
    dml_lplr_conf = DoubleMLLPLR(dml_data, ml_M, ml_t, ml_m)
    msg = r"Apply fit\(\) before confint\(\)."
    with pytest.raises(ValueError, match=msg):
        dml_lplr_conf.confint()
    dml_lplr_conf.fit()

    msg = "joint must be True or False. Got 1."
    with pytest.raises(TypeError, match=msg):
        dml_lplr_conf.confint(joint=1)
    msg = "The confidence level must be of float type. 5% of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_lplr_conf.confint(level="5%")
    msg = r"The confidence level must be in \(0,1\). 0.0 was passed."
    with pytest.raises(ValueError, match=msg):
        dml_lplr_conf.confint(level=0.0)

    msg = r"Apply bootstrap\(\) before confint\(joint=True\)."
    with pytest.raises(ValueError, match=msg):
        dml_lplr_conf.confint(joint=True)
    dml_lplr_conf.bootstrap()
    df_lplr_ci = dml_lplr_conf.confint(joint=True)
    assert isinstance(df_lplr_ci, pd.DataFrame)


@pytest.mark.ci
def test_lplr_exception_set_ml_nuisance_params():
    # invalid learner name
    msg = "Invalid nuisance learner g. Valid nuisance learner ml_m or ml_a or ml_t or ml_M.*"
    with pytest.raises(ValueError, match=msg):
        dml_lplr.set_ml_nuisance_params("g", "d", {"alpha": 0.1})
    # invalid treatment variable
    msg = "Invalid treatment variable y. Valid treatment variable d."
    with pytest.raises(ValueError, match=msg):
        dml_lplr.set_ml_nuisance_params("ml_M", "y", {"alpha": 0.1})


class _DummyNoSetParams:
    def fit(self):
        pass


class _DummyNoGetParams(_DummyNoSetParams):
    def set_params(self):
        pass


class _DummyNoClassifier(_DummyNoGetParams):
    def get_params(self):
        pass

    def predict(self):
        pass


class LogisticRegressionManipulatedType(LogisticRegression):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = None
        return tags


@pytest.mark.ci
@pytest.mark.filterwarnings(
    r"ignore:.*is \(probably\) neither a regressor nor a classifier.*:UserWarning",
)
def test_lplr_exception_learner():
    err_msg_prefix = "Invalid learner provided for ml_t: "

    msg = err_msg_prefix + "provide an instance of a learner instead of a class."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, Lasso, ml_m)
    msg = err_msg_prefix + r"BaseEstimator\(\) has no method .fit\(\)."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, BaseEstimator(), ml_m)
    msg = r"has no method .set_params\(\)."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, _DummyNoSetParams(), ml_m)
    msg = r"has no method .get_params\(\)."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, _DummyNoGetParams(), ml_m)

    # ml_m may not be a classifier when treatment is not binary
    msg = (
        r"The ml_m learner LogisticRegression\(\) was identified as classifier "
        r"but at least one treatment variable is not binary with values 0 and 1\."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, ml_t, LogisticRegression())

    # ml_m may not be a classifier when treatment is not binary
    msg = (
        r"The ml_a learner LogisticRegression\(\) was identified as classifier "
        r"but at least one treatment variable is not binary with values 0 and 1\."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, ml_t, ml_m, ml_a=LogisticRegression())

    # ml_m may not be a classifier when treatment is not binary
    dml_data_binary = make_lplr_LZZ2020(treatment="binary")
    msg = 'Learner "ml_a" who supports sample_weight is required for score type "instrument"'
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLLPLR(dml_data_binary, ml_M, ml_t, ml_m, ml_a=LabelSpreading(), score="instrument")

    # construct a classifier which is not identifiable as classifier via is_classifier by sklearn
    log_reg = LogisticRegressionManipulatedType()
    msg = (
        r"Learner provided for ml_m is probably invalid: LogisticRegressionManipulatedType\(\) is \(probably\) "
        r"neither a regressor nor a classifier. Method predict is used for prediction\."
    )
    with pytest.warns(UserWarning, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, ml_t, log_reg)


@pytest.mark.ci
@pytest.mark.filterwarnings(
    r"ignore:.*is \(probably\) neither a regressor nor a classifier.*:UserWarning",
    r"ignore: Learner provided for ml_m is probably invalid.*is \(probably\) no classifier.*:UserWarning",
)
def test_lplr_exception_and_warning_learner():
    # invalid ml_M (must be a classifier with predict_proba)
    with pytest.raises(TypeError):
        _ = DoubleMLLPLR(dml_data, _DummyNoClassifier(), ml_t, ml_m)
    msg = "Invalid learner provided for ml_M: " + r"Lasso\(\) has no method .predict_proba\(\)."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLLPLR(dml_data, Lasso(), ml_t, ml_m)
    msg = (
        r"The ml_m learner RandomForestRegressor\(.*\) was identified as regressor but at least one treatment "
        r"variable is binary with values 0 and 1."
    )
    with pytest.warns(match=msg):
        _ = DoubleMLLPLR(dml_data_binary, ml_M, ml_t, ml_m)
    msg = (
        r"The ml_a learner RandomForestRegressor\(.*\) was identified as regressor but at least one treatment "
        r"variable is binary with values 0 and 1."
    )
    with pytest.warns(match=msg):
        _ = DoubleMLLPLR(dml_data_binary, ml_M, ml_t, ml_M, ml_a=ml_m)


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


# Classifier that returns hard labels (0/1) via predict_proba to trigger the binary-predictions error
class HardLabelPredictProba(LogisticRegression):
    def predict_proba(self, X):
        labels = super().predict(X).astype(int)
        return np.column_stack((1 - labels, labels))


@pytest.mark.ci
def test_lplr_nan_prediction():
    msg = r"Predictions from learner LassoWithNanPred\(\) for ml_t are not finite."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, LassoWithNanPred(), ml_m).fit()
    msg = r"Predictions from learner LassoWithInfPred\(\) for ml_t are not finite."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLLPLR(dml_data, ml_M, LassoWithInfPred(), ml_m).fit()


@pytest.mark.ci
def test_double_ml_exception_evaluate_learner():
    dml_lplr_obj = DoubleMLLPLR(
        dml_data,
        ml_M=LogisticRegression(),
        ml_t=Lasso(),
        ml_m=RandomForestRegressor(),
        n_folds=5,
        score="nuisance_space",
    )

    msg = r"Apply fit\(\) before evaluate_learners\(\)."
    with pytest.raises(ValueError, match=msg):
        dml_lplr_obj.evaluate_learners()

    dml_lplr_obj.fit()

    msg = "metric should be a callable. 'mse' was passed."
    with pytest.raises(TypeError, match=msg):
        dml_lplr_obj.evaluate_learners(metric="mse")

    msg = (
        r"The learners have to be a subset of \['ml_m', 'ml_a', 'ml_t', 'ml_M'.*\]\. "
        r"Learners \['ml_mu', 'ml_p'\] provided."
    )
    with pytest.raises(ValueError, match=msg):
        dml_lplr_obj.evaluate_learners(learners=["ml_mu", "ml_p"])

    def eval_fct(y_pred, y_true):
        return np.nan

    with pytest.raises(ValueError):
        dml_lplr_obj.evaluate_learners(metric=eval_fct)


@pytest.mark.ci
def test_lplr_exception_binary_predictions_from_classifier():
    # Expect error because ml_m returns binary labels instead of probabilities for a binary treatment
    msg = (
        r"For the binary treatment variable d, predictions obtained with the ml_m learner "
        r"HardLabelPredictProba\(\) are also observed to be binary with values 0 and 1\. "
        r"Make sure that for classifiers probabilities and not labels are predicted\."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLLPLR(dml_data_binary, ml_M, ml_t, HardLabelPredictProba()).fit()
