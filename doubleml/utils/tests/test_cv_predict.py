import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from doubleml.utils._estimation import _double_dml_cv_predict


@pytest.mark.ci
def test_double_dml_cv_predict_uses_inner_fold_count():
    X = np.arange(12).reshape(-1, 1)
    y = np.zeros(12)

    smpls = [
        (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10, 11])),
        (np.array([6, 7, 8, 9, 10, 11]), np.array([0, 1, 2, 3, 4, 5])),
    ]

    smpls_inner = [
        [
            (np.array([0, 1, 2]), np.array([3, 4])),
            (np.array([3, 4, 5]), np.array([0, 1])),
            (np.array([0, 1, 5]), np.array([2, 3])),
        ],
        [
            (np.array([6, 7]), np.array([8, 9])),
            (np.array([8, 9, 10]), np.array([6, 7])),
        ],
    ]

    out = _double_dml_cv_predict(
        DummyRegressor(strategy="constant", constant=7.0),
        "ml_M",
        X,
        y,
        smpls=smpls,
        smpls_inner=smpls_inner,
    )

    assert np.allclose(out["preds"][smpls[0][1]], 7.0)
    assert np.allclose(out["preds"][smpls[1][1]], 7.0)


@pytest.mark.ci
def test_double_dml_cv_predict_fold_specific_params():
    """Outer-fold parameters are applied to all inner models of that fold, independent of the inner fold count."""
    X = np.arange(12).reshape(-1, 1)
    y = np.zeros(12)

    smpls = [
        (np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10, 11])),
        (np.array([6, 7, 8, 9, 10, 11]), np.array([0, 1, 2, 3, 4, 5])),
    ]

    smpls_inner = [
        [
            (np.array([0, 1, 2]), np.array([3, 4])),
            (np.array([3, 4, 5]), np.array([0, 1])),
            (np.array([0, 1, 5]), np.array([2, 3])),
        ],
        [
            (np.array([6, 7]), np.array([8, 9])),
            (np.array([8, 9, 10]), np.array([6, 7])),
        ],
    ]

    out = _double_dml_cv_predict(
        DummyRegressor(strategy="constant", constant=7.0),
        "ml_M",
        X,
        y,
        smpls=smpls,
        smpls_inner=smpls_inner,
        est_params=[{"constant": 1.0}, {"constant": 2.0}],
    )

    assert np.allclose(out["preds"][smpls[0][1]], 1.0)
    assert np.allclose(out["preds"][smpls[1][1]], 2.0)
    for i_fold, constant in enumerate([1.0, 2.0]):
        inner_test = np.concatenate([test for _, test in smpls_inner[i_fold]])
        assert np.allclose(out["preds_inner"][i_fold][inner_test], constant)
