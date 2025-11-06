import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml.did import DoubleMLDIDBinary
from doubleml.utils.propensity_score_processing import PSProcessorConfig


@pytest.mark.ci
@pytest.mark.parametrize(
    "ps_config",
    [
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method=None, cv_calibration=False),
        PSProcessorConfig(clipping_threshold=0.05, calibration_method=None, cv_calibration=False),
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method="isotonic", cv_calibration=False),
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method="isotonic", cv_calibration=True),
    ],
)
def test_did_binary_ml_m_predictions_ps_processor(generate_data_did_binary, ps_config):
    dml_data = generate_data_did_binary
    np.random.seed(3141)
    dml_did = DoubleMLDIDBinary(
        obj_dml_data=dml_data,
        g_value=1,
        t_value_pre=0,
        t_value_eval=1,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(),
        ps_processor_config=ps_config,
        n_rep=1,
        score="observational",
    )
    dml_did.fit(store_predictions=True)
    ml_m_preds = dml_did.predictions["ml_m"][:, 0, 0]
    # Just check that predictions are within [clipping_threshold, 1-clipping_threshold]
    assert np.all(ml_m_preds >= ps_config.clipping_threshold)
    assert np.all(ml_m_preds <= 1 - ps_config.clipping_threshold)


@pytest.mark.ci
def test_did_binary_ml_m_predictions_ps_processor_differences(generate_data_did_binary):
    dml_data = generate_data_did_binary
    np.random.seed(3141)
    configs = [
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method=None, cv_calibration=False),
        PSProcessorConfig(clipping_threshold=0.05, calibration_method=None, cv_calibration=False),
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method="isotonic", cv_calibration=False),
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method="isotonic", cv_calibration=True),
    ]
    preds = []
    for cfg in configs:
        dml_did = DoubleMLDIDBinary(
            obj_dml_data=dml_data,
            g_value=1,
            t_value_pre=0,
            t_value_eval=1,
            ml_g=LinearRegression(),
            ml_m=LogisticRegression(),
            ps_processor_config=cfg,
            n_rep=1,
            score="observational",
        )
        dml_did.fit(store_predictions=True)
        preds.append(dml_did.predictions["ml_m"][:, 0, 0])
    # Check that at least two configurations yield different predictions (element-wise)
    diffs = [not np.allclose(preds[i], preds[j], atol=1e-6) for i in range(len(preds)) for j in range(i + 1, len(preds))]
    assert any(diffs)
