import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import doubleml as dml
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
def test_lpq_ml_m_predictions_ps_processor(generate_data_local_quantiles, ps_config):
    x, y, d, z = generate_data_local_quantiles
    dml_data = dml.DoubleMLData.from_arrays(x=x, y=y, d=d, z=z)
    np.random.seed(3141)
    dml_lpq = dml.DoubleMLLPQ(
        obj_dml_data=dml_data,
        ml_g=LogisticRegression(),
        ml_m=LogisticRegression(),
        treatment=1,
        quantile=0.5,
        ps_processor_config=ps_config,
        n_rep=1,
    )
    dml_lpq.fit(store_predictions=True)
    ml_m_preds = dml_lpq.predictions["ml_m_z"][:, 0, 0]
    # Just check that predictions are within [clipping_threshold, 1-clipping_threshold]
    assert np.all(ml_m_preds >= ps_config.clipping_threshold)
    assert np.all(ml_m_preds <= 1 - ps_config.clipping_threshold)


@pytest.mark.ci
def test_lpq_ml_m_predictions_ps_processor_differences(generate_data_local_quantiles):
    x, y, d, z = generate_data_local_quantiles
    dml_data = dml.DoubleMLData.from_arrays(x=x, y=y, d=d, z=z)
    np.random.seed(3141)
    configs = [
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method=None, cv_calibration=False),
        PSProcessorConfig(clipping_threshold=0.05, calibration_method=None, cv_calibration=False),
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method="isotonic", cv_calibration=False),
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method="isotonic", cv_calibration=True),
    ]
    preds = []
    for cfg in configs:
        dml_lpq = dml.DoubleMLLPQ(
            obj_dml_data=dml_data,
            ml_g=LogisticRegression(),
            ml_m=LogisticRegression(),
            treatment=1,
            quantile=0.5,
            ps_processor_config=cfg,
            n_rep=1,
        )
        dml_lpq.fit(store_predictions=True)
        preds.append(dml_lpq.predictions["ml_m_z"][:, 0, 0])
    # Check that at least two configurations yield different predictions (element-wise)
    diffs = [not np.allclose(preds[i], preds[j], atol=1e-6) for i in range(len(preds)) for j in range(i + 1, len(preds))]
    assert any(diffs)
