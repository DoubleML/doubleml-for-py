"""Test RDFlex with a Pipeline learner that receives the sample weights via scikit-learn metadata routing."""

import numpy as np
import pytest
from sklearn import config_context
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from doubleml import DoubleMLRDDData
from doubleml.rdd import RDFlex
from doubleml.rdd.datasets import make_simple_rdd_data


@pytest.fixture(scope="module", params=[False, True])
def fuzzy(request):
    return request.param


@pytest.mark.ci_rdd
def test_rdd_pipeline_metadata_routing(fuzzy):
    """With metadata routing enabled and weights requested on all steps, RDFlex fits Pipeline learners."""
    np.random.seed(3141)
    data = make_simple_rdd_data(n_obs=500, fuzzy=fuzzy)
    dml_data = DoubleMLRDDData.from_arrays(x=data["X"], y=data["Y"], d=data["D"], score=data["score"])
    with config_context(enable_metadata_routing=True):
        ml_g = make_pipeline(StandardScaler().set_fit_request(sample_weight=True), Lasso().set_fit_request(sample_weight=True))
        ml_m = None
        if fuzzy:
            ml_m = make_pipeline(
                StandardScaler().set_fit_request(sample_weight=True),
                LogisticRegression().set_fit_request(sample_weight=True),
            )
        rdd_model = RDFlex(dml_data, ml_g=ml_g, ml_m=ml_m, fuzzy=fuzzy).fit()
    assert np.all(np.isfinite(rdd_model.coef))
    assert np.all(rdd_model.se > 0)
