import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from doubleml import DoubleMLData
from doubleml.irm.iivm import DoubleMLIIVM
from doubleml.utils.propensity_score_processing import PSProcessorConfig


@pytest.fixture
def dml_data_iivm(generate_data_iivm):
    data = generate_data_iivm
    x_cols = data.columns[data.columns.str.startswith("X")].tolist()
    dml_data = DoubleMLData(data, "y", ["d"], x_cols, "z")
    return dml_data


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
def test_iivm_ml_m_predictions_ps_processor(dml_data_iivm, ps_config):
    np.random.seed(3141)
    dml_iivm = DoubleMLIIVM(
        obj_dml_data=dml_data_iivm,
        ml_g=LinearRegression(),
        ml_m=LogisticRegression(),
        ml_r=LogisticRegression(),
        ps_processor_config=ps_config,
        n_rep=1,
    )
    dml_iivm.fit(store_predictions=True)
    ml_m_preds = dml_iivm.predictions["ml_m"][:, 0, 0]
    # Just check that predictions are within [clipping_threshold, 1-clipping_threshold]
    assert np.all(ml_m_preds >= ps_config.clipping_threshold)
    assert np.all(ml_m_preds <= 1 - ps_config.clipping_threshold)


@pytest.mark.ci
def test_iivm_ml_m_predictions_ps_processor_differences(dml_data_iivm):
    np.random.seed(3141)
    configs = [
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method=None, cv_calibration=False),
        PSProcessorConfig(clipping_threshold=0.05, calibration_method=None, cv_calibration=False),
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method="isotonic", cv_calibration=False),
        PSProcessorConfig(clipping_threshold=1e-2, calibration_method="isotonic", cv_calibration=True),
    ]
    preds = []
    for cfg in configs:
        dml_iivm = DoubleMLIIVM(
            obj_dml_data=dml_data_iivm,
            ml_g=LinearRegression(),
            ml_m=LogisticRegression(),
            ml_r=LogisticRegression(),
            ps_processor_config=cfg,
            n_rep=1,
        )
        dml_iivm.fit(store_predictions=True)
        preds.append(dml_iivm.predictions["ml_m"][:, 0, 0])
    # Check that at least two configurations yield different predictions (element-wise)
    diffs = [not np.allclose(preds[i], preds[j], atol=1e-6) for i in range(len(preds)) for j in range(i + 1, len(preds))]
    assert any(diffs)
