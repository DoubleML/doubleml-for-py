import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso, LogisticRegression

from doubleml import DoubleMLPanelData, DoubleMLPLPR
from doubleml.plm.datasets import make_plpr_CP2025

np.random.seed(3141)
num_id = 100
# create test data and basic learners
plpr_data = make_plpr_CP2025(num_id=num_id, theta=0.5, dim_x=30)
plpr_data_binary = plpr_data.copy()
plpr_data_binary["d"] = np.where(plpr_data_binary["d"] > 0, 1, 0)

x_cols = [col for col in plpr_data.columns if "x" in col]
dml_data = DoubleMLPanelData(
    plpr_data,
    y_col="y",
    d_cols="d",
    t_col="time",
    id_col="id",
    static_panel=True,
)
dml_data_iv = DoubleMLPanelData(
    plpr_data,
    y_col="y",
    d_cols="d",
    t_col="time",
    id_col="id",
    x_cols=x_cols[:-1],
    z_cols=x_cols[-1],
    static_panel=True,
)
dml_data_binary = DoubleMLPanelData(
    plpr_data_binary,
    y_col="y",
    d_cols="d",
    t_col="time",
    id_col="id",
    static_panel=True,
)
ml_l = Lasso(alpha=0.1)
ml_m = Lasso(alpha=0.1)
ml_g = Lasso(alpha=0.1)
dml_plpr = DoubleMLPLPR(dml_data, ml_l, ml_m)
dml_plpr_iv_type = DoubleMLPLPR(dml_data, ml_l, ml_m, ml_g, score="IV-type")


@pytest.mark.ci
def test_plpr_exception_data():
    msg = "The data must be of DoubleMLPanelData type. <class 'pandas.core.frame.DataFrame'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLPR(pd.DataFrame(), ml_l, ml_m)

    # instrument
    msg = (
        r"Incompatible data. x30 have been set as instrumental variable\(s\). "
        "DoubleMLPLPR currently does not support instrumental variables."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLPR(dml_data_iv, ml_l, ml_m)


@pytest.mark.ci
def test_plpr_exception_scores():
    msg = "Invalid score IV. Valid score IV-type or partialling out."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLPR(dml_data, ml_l, ml_m, score="IV")
    msg = "score should be either a string or a callable. 0 was passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLPR(dml_data, ml_l, ml_m, score=0)


@pytest.mark.ci
def test_plpr_exception_approach():
    # PLPR valid approaches are 'cre_general', 'cre_normal', 'fd_exact', and 'wg_approx'
    msg = "Invalid approach cre. Valid approach cre_general or cre_normal or fd_exact or wg_approx."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLPR(dml_data, ml_l, ml_m, approach="cre")
    msg = "approach should be a string. 4 was passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLPR(dml_data, ml_l, ml_m, approach=4)


@pytest.mark.ci
def test_plpr_exception_resampling():
    msg = "The number of folds must be of int type. 1.5 of type <class 'float'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLPR(dml_data, ml_l, ml_m, n_folds=1.5)
    msg = "The number of repetitions for the sample splitting must be of int type. 1.5 of type <class 'float'> was passed."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLPR(dml_data, ml_l, ml_m, n_rep=1.5)
    msg = "The number of folds must be positive. 0 was passed."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLPR(dml_data, ml_l, ml_m, n_folds=0)
    msg = "The number of repetitions for the sample splitting must be positive. 0 was passed."
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLPR(dml_data, ml_l, ml_m, n_rep=0)
    msg = "draw_sample_splitting must be True or False. Got true."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLPR(dml_data, ml_l, ml_m, draw_sample_splitting="true")


@pytest.mark.ci
def test_plpr_exception_get_params():
    msg = "Invalid nuisance learner ml_r. Valid nuisance learner ml_l or ml_m."
    with pytest.raises(ValueError, match=msg):
        dml_plpr.get_params("ml_r")
    msg = "Invalid nuisance learner ml_g. Valid nuisance learner ml_l or ml_m."
    with pytest.raises(ValueError, match=msg):
        dml_plpr.get_params("ml_g")
    msg = "Invalid nuisance learner ml_r. Valid nuisance learner ml_l or ml_m or ml_g."
    with pytest.raises(ValueError, match=msg):
        dml_plpr_iv_type.get_params("ml_r")


# TODO: test_doubleml_exception_onefold(): for plpr?
@pytest.mark.ci
def test_plpr_exception_smpls():
    msg = (
        "Sample splitting not specified. "
        r"Either draw samples via .draw_sample splitting\(\) or set external samples via .set_sample_splitting\(\)."
    )
    dml_plpr_no_smpls = DoubleMLPLPR(dml_data, ml_l, ml_m, draw_sample_splitting=False)
    with pytest.raises(ValueError, match=msg):
        _ = dml_plpr_no_smpls.smpls

    dml_plpr_cluster = dml_plpr
    smpls = dml_plpr.smpls
    msg = "For cluster data, all_smpls_cluster must be provided."
    with pytest.raises(ValueError, match=msg):
        _ = dml_plpr_cluster.set_sample_splitting(smpls)

    all_smpls_cluster = copy.deepcopy(dml_plpr_cluster.smpls_cluster)
    all_smpls_cluster.append(all_smpls_cluster[0])
    msg = "Invalid samples provided. Number of repetitions for all_smpls and all_smpls_cluster must be the same."
    with pytest.raises(ValueError, match=msg):
        _ = dml_plpr_cluster.set_sample_splitting(all_smpls=dml_plpr_cluster.smpls, all_smpls_cluster=all_smpls_cluster)

    all_smpls_cluster = copy.deepcopy(dml_plpr_cluster.smpls_cluster)
    all_smpls_cluster.append(all_smpls_cluster[0])
    msg = "Invalid samples provided. Number of repetitions for all_smpls and all_smpls_cluster must be the same."
    with pytest.raises(ValueError, match=msg):
        _ = dml_plpr_cluster.set_sample_splitting(all_smpls=dml_plpr_cluster.smpls, all_smpls_cluster=all_smpls_cluster)

    all_smpls_cluster = copy.deepcopy(dml_plpr_cluster.smpls_cluster)
    all_smpls_cluster[0][0][1][0] = np.append(all_smpls_cluster[0][0][1][0], [11], axis=0)
    msg = "Invalid cluster partition provided. At least one inner list does not form a partition."
    with pytest.raises(ValueError, match=msg):
        _ = dml_plpr_cluster.set_sample_splitting(all_smpls=dml_plpr_cluster.smpls, all_smpls_cluster=all_smpls_cluster)

    all_smpls_cluster = copy.deepcopy(dml_plpr_cluster.smpls_cluster)
    all_smpls_cluster[0][0][1][0][1] = 11
    msg = "Invalid cluster partition provided. At least one inner list does not form a partition."
    with pytest.raises(ValueError, match=msg):
        _ = dml_plpr_cluster.set_sample_splitting(all_smpls=dml_plpr_cluster.smpls, all_smpls_cluster=all_smpls_cluster)


@pytest.mark.ci
def test_plpr_exception_fit():
    msg = "The number of CPUs used to fit the learners must be of int type. 5 of type <class 'str'> was passed."
    with pytest.raises(TypeError, match=msg):
        dml_plpr.fit(n_jobs_cv="5")
    msg = "store_predictions must be True or False. Got 1."
    with pytest.raises(TypeError, match=msg):
        dml_plpr.fit(store_predictions=1)
    msg = "store_models must be True or False. Got 1."
    with pytest.raises(TypeError, match=msg):
        dml_plpr.fit(store_models=1)


@pytest.mark.ci
def test_plpr_exception_set_ml_nuisance_params():
    msg = "Invalid nuisance learner g. Valid nuisance learner ml_l or ml_m."
    with pytest.raises(ValueError, match=msg):
        dml_plpr.set_ml_nuisance_params("g", "d", {"alpha": 0.1})
    msg = "Invalid treatment variable y. Valid treatment variable d_diff."
    with pytest.raises(ValueError, match=msg):
        dml_plpr.set_ml_nuisance_params("ml_l", "y", {"alpha": 0.1})


class _DummyNoSetParams:
    def fit(self):
        pass


class _DummyNoGetParams(_DummyNoSetParams):
    def set_params(self):
        pass


class _DummyNoClassifier(_DummyNoGetParams, BaseEstimator):
    def get_params(self, deep=True):
        return {}

    def predict_proba(self):
        pass


class LogisticRegressionManipulatedPredict(LogisticRegression):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = None
        return tags

    def predict(self, X):
        if self.max_iter == 314:
            preds = super().predict_proba(X)[:, 1]
        else:
            preds = super().predict(X)
        return preds


@pytest.mark.ci
def test_plpr_exception_learner():
    err_msg_prefix = "Invalid learner provided for ml_l: "

    msg = err_msg_prefix + "provide an instance of a learner instead of a class."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLPR(dml_data, Lasso, ml_m)
    msg = err_msg_prefix + r"BaseEstimator\(\) has no method .fit\(\)."
    with pytest.raises(TypeError, match=msg):
        _ = DoubleMLPLPR(dml_data, BaseEstimator(), ml_m)
    # msg = err_msg_prefix + r'_DummyNoSetParams\(\) has no method .set_params\(\).'
    with pytest.raises(TypeError):
        _ = DoubleMLPLPR(dml_data, _DummyNoSetParams(), ml_m)
    # msg = err_msg_prefix + r'_DummyNoSetParams\(\) has no method .get_params\(\).'
    with pytest.raises(TypeError):
        _ = DoubleMLPLPR(dml_data, _DummyNoGetParams(), ml_m)

    # we allow classifiers for ml_m in PLPR, but only for binary treatment variables
    msg = (
        r"The ml_m learner LogisticRegression\(\) was identified as classifier "
        "but at least one treatment variable is not binary with values 0 and 1."
    )
    with pytest.raises(ValueError, match=msg):
        _ = DoubleMLPLPR(dml_data, Lasso(), LogisticRegression())

    msg = r"For score = 'IV-type', learners ml_l and ml_g should be specified. Set ml_g = clone\(ml_l\)."
    with pytest.warns(UserWarning, match=msg):
        _ = DoubleMLPLPR(dml_data, ml_l=Lasso(), ml_m=ml_m, score="IV-type")

    msg = 'A learner ml_g has been provided for score = "partialling out" but will be ignored.'
    with pytest.warns(UserWarning, match=msg):
        _ = DoubleMLPLPR(dml_data, ml_l=Lasso(), ml_m=Lasso(), ml_g=Lasso(), score="partialling out")

    # construct a classifier which is not identifiable as classifier via is_classifier by sklearn
    # it then predicts labels and therefore an exception will be thrown
    # TODO: cases with approaches cre_general, fd_exact, wg_approx
    log_reg = LogisticRegressionManipulatedPredict()
    msg_warn = (
        r"Learner provided for ml_m is probably invalid: LogisticRegressionManipulatedPredict\(\) is \(probably\) "
        "neither a regressor nor a classifier. Method predict is used for prediction."
    )
    with pytest.warns(UserWarning, match=msg_warn):
        dml_plpr_hidden_classifier = DoubleMLPLPR(dml_data_binary, Lasso(), log_reg, approach="cre_normal")
    msg = (
        r"For the binary variable d, predictions obtained with the ml_m learner LogisticRegressionManipulatedPredict\(\) "
        "are also observed to be binary with values 0 and 1. Make sure that for classifiers probabilities and not "
        "labels are predicted."
    )
    with pytest.warns(UserWarning, match=msg_warn):
        with pytest.raises(ValueError, match=msg):
            dml_plpr_hidden_classifier.fit()


@pytest.mark.ci
@pytest.mark.filterwarnings("ignore:Learner provided for")
def test_plpr_exception_and_warning_learner():
    # msg = err_msg_prefix + r'_DummyNoClassifier\(\) has no method .predict\(\).'
    with pytest.raises(TypeError):
        _ = DoubleMLPLPR(dml_data, _DummyNoClassifier(), Lasso())
