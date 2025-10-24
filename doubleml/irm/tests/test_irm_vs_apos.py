import copy

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import doubleml as dml
from doubleml.utils._propensity_score import _propensity_score_adjustment
from doubleml.utils.propensity_score_processing import PSProcessorConfig


@pytest.fixture(
    scope="module",
    params=[
        [LinearRegression(), LogisticRegression(solver="lbfgs", max_iter=250)],
        [
            RandomForestRegressor(max_depth=5, n_estimators=10, random_state=42),
            RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42),
        ],
    ],
)
def learner(request):
    return request.param


@pytest.fixture(scope="module", params=[1, 3])
def n_rep(request):
    return request.param


@pytest.fixture(scope="module", params=[False, True])
def normalize_ipw(request):
    return request.param


@pytest.fixture(scope="module", params=[0.2, 0.15])
def clipping_threshold(request):
    return request.param


@pytest.fixture(scope="module")
def dml_irm_apos_fixture(generate_data_irm, learner, n_rep, normalize_ipw, clipping_threshold):

    # collect data
    (x, y, d) = generate_data_irm
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)

    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

    n_folds = 5
    kwargs = {
        "n_folds": n_folds,
        "n_rep": n_rep,
        "ps_processor_config": PSProcessorConfig(clipping_threshold=clipping_threshold),
        "normalize_ipw": normalize_ipw,
    }

    dml_irm = dml.DoubleMLIRM(
        obj_dml_data=obj_dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        score="ATE",
        **kwargs,
    )
    dml_irm.fit()

    m_hat = dml_irm.predictions["ml_m"][:, :, 0]
    g0_hat = dml_irm.predictions["ml_g0"][:, :, 0]
    g1_hat = dml_irm.predictions["ml_g1"][:, :, 0]

    external_predictions_apos = {
        0: {
            "ml_m": 1.0 - m_hat,
            "ml_g_d_lvl1": g0_hat,
            "ml_g_d_lvl0": g1_hat,
        },
        1: {
            "ml_m": m_hat,
            "ml_g_d_lvl1": g1_hat,
            "ml_g_d_lvl0": g0_hat,
        },
    }

    dml_apos = dml.DoubleMLAPOS(obj_dml_data=obj_dml_data, ml_g=ml_g, ml_m=ml_m, treatment_levels=[0, 1], **kwargs)
    dml_apos = dml_apos.fit(external_predictions=external_predictions_apos)
    causal_contrast = dml_apos.causal_contrast(reference_levels=[0])

    irm_confint = dml_irm.confint().values
    causal_contrast_confint = causal_contrast.confint().values

    # sensitivity analysis
    dml_irm.sensitivity_analysis()
    causal_contrast.sensitivity_analysis()

    result_dict = {
        "dml_irm": dml_irm,
        "dml_apos": dml_apos,
        "causal_contrast": causal_contrast,
        "irm_confint": irm_confint,
        "causal_contrast_confint": causal_contrast_confint,
    }
    return result_dict


@pytest.mark.ci
def test_apos_vs_irm_thetas(dml_irm_apos_fixture):
    assert np.allclose(
        dml_irm_apos_fixture["dml_irm"].framework.all_thetas,
        dml_irm_apos_fixture["causal_contrast"].all_thetas,
        rtol=1e-9,
        atol=1e-4,
    )


@pytest.mark.ci
def test_apos_vs_irm_ses(dml_irm_apos_fixture):
    assert np.allclose(
        dml_irm_apos_fixture["dml_irm"].framework.all_ses,
        dml_irm_apos_fixture["causal_contrast"].all_ses,
        rtol=1e-9,
        atol=1e-4,
    )


@pytest.mark.ci
def test_apos_vs_irm_confint(dml_irm_apos_fixture):
    assert np.allclose(
        dml_irm_apos_fixture["irm_confint"],
        dml_irm_apos_fixture["causal_contrast_confint"],
        rtol=1e-9,
        atol=1e-4,
    )


@pytest.mark.ci
def test_apos_vs_irm_sensitivity(dml_irm_apos_fixture):
    params_irm = dml_irm_apos_fixture["dml_irm"].sensitivity_params
    params_causal_contrast = dml_irm_apos_fixture["causal_contrast"].sensitivity_params

    for key in ["theta", "se", "ci"]:
        for boundary in ["upper", "lower"]:
            assert np.allclose(
                params_irm[key][boundary],
                params_causal_contrast[key][boundary],
                rtol=1e-9,
                atol=1e-4,
            )

    for key in ["rv", "rva"]:
        assert np.allclose(
            params_irm[key],
            params_causal_contrast[key],
            rtol=1e-9,
            atol=1e-4,
        )


@pytest.fixture(scope="module")
def dml_irm_apos_weighted_fixture(generate_data_irm, learner, n_rep, normalize_ipw, clipping_threshold):

    # collect data
    (x, y, d) = generate_data_irm
    obj_dml_data = dml.DoubleMLData.from_arrays(x, y, d)

    # Set machine learning methods for m & g
    ml_g = clone(learner[0])
    ml_m = clone(learner[1])

    n_folds = 5
    kwargs = {
        "n_folds": n_folds,
        "n_rep": n_rep,
        "ps_processor_config": PSProcessorConfig(clipping_threshold=clipping_threshold),
        "normalize_ipw": normalize_ipw,
    }

    dml_irm = dml.DoubleMLIRM(
        obj_dml_data=obj_dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        score="ATTE",
        **kwargs,
    )
    dml_irm.fit()

    m_hat = dml_irm.predictions["ml_m"][:, :, 0]
    g0_hat = dml_irm.predictions["ml_g0"][:, :, 0]
    g1_hat = dml_irm.predictions["ml_g1"][:, :, 0]

    # define weights
    p_hat = np.mean(d)
    m_hat_adjusted = copy.deepcopy(m_hat)
    for i_rep in range(n_rep):
        m_hat_adjusted[:, i_rep] = _propensity_score_adjustment(
            propensity_score=m_hat[:, i_rep], treatment_indicator=d, normalize_ipw=normalize_ipw
        )
    weights_dict = {
        "weights": d / p_hat,
        "weights_bar": m_hat_adjusted / p_hat,
    }

    external_predictions_irm = {
        "d": {
            "ml_m": m_hat,
            "ml_g1": g1_hat,
            "ml_g0": g0_hat,
        }
    }
    dml_irm_weighted = dml.DoubleMLIRM(
        obj_dml_data=obj_dml_data,
        ml_g=ml_g,
        ml_m=ml_m,
        score="ATE",
        weights=weights_dict,
        **kwargs,
    )
    dml_irm_weighted.fit(external_predictions=external_predictions_irm)

    external_predictions_apos = {
        0: {
            "ml_m": 1.0 - m_hat,
            "ml_g_d_lvl1": g0_hat,
            "ml_g_d_lvl0": g1_hat,
        },
        1: {
            "ml_m": m_hat,
            "ml_g_d_lvl1": g1_hat,
            "ml_g_d_lvl0": g0_hat,
        },
    }

    dml_apos = dml.DoubleMLAPOS(
        obj_dml_data=obj_dml_data, ml_g=ml_g, ml_m=ml_m, treatment_levels=[0, 1], weights=weights_dict, **kwargs
    )
    dml_apos = dml_apos.fit(external_predictions=external_predictions_apos)
    causal_contrast = dml_apos.causal_contrast(reference_levels=[0])

    irm_confint = dml_irm.confint().values
    irm_weighted_confint = dml_irm_weighted.confint().values
    causal_contrast_confint = causal_contrast.confint().values

    # sensitivity analysis
    dml_irm.sensitivity_analysis()
    dml_irm_weighted.sensitivity_analysis()
    causal_contrast.sensitivity_analysis()

    result_dict = {
        "dml_irm": dml_irm,
        "dml_irm_weighted": dml_irm_weighted,
        "dml_apos": dml_apos,
        "causal_contrast": causal_contrast,
        "irm_confint": irm_confint,
        "irm_weighted_confint": irm_weighted_confint,
        "causal_contrast_confint": causal_contrast_confint,
    }
    return result_dict


@pytest.mark.ci
def test_apos_vs_irm_weighted_thetas(dml_irm_apos_weighted_fixture):
    assert np.allclose(
        dml_irm_apos_weighted_fixture["dml_irm"].framework.all_thetas,
        dml_irm_apos_weighted_fixture["dml_irm_weighted"].framework.all_thetas,
        rtol=1e-9,
        atol=1e-4,
    )

    assert np.allclose(
        dml_irm_apos_weighted_fixture["dml_irm_weighted"].framework.all_thetas,
        dml_irm_apos_weighted_fixture["causal_contrast"].all_thetas,
        rtol=1e-9,
        atol=1e-4,
    )


@pytest.mark.ci
def test_apos_vs_irm_weighted_ses(dml_irm_apos_weighted_fixture):
    assert np.allclose(
        dml_irm_apos_weighted_fixture["dml_irm"].framework.all_ses,
        dml_irm_apos_weighted_fixture["dml_irm_weighted"].framework.all_ses,
        rtol=1e-9,
        atol=1e-4,
    )

    assert np.allclose(
        dml_irm_apos_weighted_fixture["dml_irm_weighted"].framework.all_ses,
        dml_irm_apos_weighted_fixture["causal_contrast"].all_ses,
        rtol=1e-9,
        atol=1e-4,
    )


@pytest.mark.ci
def test_apos_vs_irm_weighted_confint(dml_irm_apos_weighted_fixture):
    assert np.allclose(
        dml_irm_apos_weighted_fixture["irm_confint"],
        dml_irm_apos_weighted_fixture["irm_weighted_confint"],
        rtol=1e-9,
        atol=1e-4,
    )

    assert np.allclose(
        dml_irm_apos_weighted_fixture["irm_weighted_confint"],
        dml_irm_apos_weighted_fixture["causal_contrast_confint"],
        rtol=1e-9,
        atol=1e-4,
    )


@pytest.mark.ci
def test_apos_vs_irm_weighted_sensitivity(dml_irm_apos_weighted_fixture):
    # TODO: Include after normalize_ipw rework, see Issue https://github.com/DoubleML/doubleml-for-py/issues/296
    params_irm = dml_irm_apos_weighted_fixture["dml_irm"].sensitivity_params
    params_irm_weighted = dml_irm_apos_weighted_fixture["dml_irm_weighted"].sensitivity_params
    params_causal_contrast = dml_irm_apos_weighted_fixture["causal_contrast"].sensitivity_params

    for key in ["theta", "se", "ci"]:
        for boundary in ["upper", "lower"]:
            # TODO: Include after normalize_ipw rework, see Issue https://github.com/DoubleML/doubleml-for-py/issues/296
            assert np.allclose(
                params_irm[key][boundary],
                params_irm_weighted[key][boundary],
                rtol=1e-9,
                atol=1e-4,
            )

            assert np.allclose(
                params_irm_weighted[key][boundary],
                params_causal_contrast[key][boundary],
                rtol=1e-9,
                atol=1e-4,
            )

    for key in ["rv", "rva"]:
        # TODO: Include after normalize_ipw rework, see Issue https://github.com/DoubleML/doubleml-for-py/issues/296
        assert np.allclose(
            params_irm[key],
            params_irm_weighted[key],
            rtol=1e-9,
            atol=1e-4,
        )

        assert np.allclose(
            params_irm_weighted[key],
            params_causal_contrast[key],
            rtol=1e-9,
            atol=1e-4,
        )
