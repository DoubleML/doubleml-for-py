import numpy as np
import pandas as pd

from doubleml.double_ml import DoubleML


def _check_basic_defaults_before_fit(dml_obj):
    # general parameters
    assert dml_obj.n_folds == 5
    assert dml_obj.n_rep == 1
    assert dml_obj.framework is None
    pd.testing.assert_frame_equal(dml_obj.summary, pd.DataFrame(columns=["coef", "std err", "t", "P>|t|"]))

    # bootstrap
    assert dml_obj.boot_method is None
    assert dml_obj.n_rep_boot is None
    assert dml_obj.boot_t_stat is None

    # sensitivity
    assert dml_obj.sensitivity_params is None
    assert dml_obj.sensitivity_elements is None


def _fit_bootstrap(dml_obj):
    dml_obj.fit()
    dml_obj.bootstrap()


def _check_basic_defaults_after_fit(dml_obj):
    # general parameters
    assert dml_obj.n_folds == 5
    assert dml_obj.n_rep == 1
    assert dml_obj.framework is not None

    # coefs and se
    assert isinstance(dml_obj.coef, np.ndarray)
    assert isinstance(dml_obj.se, np.ndarray)
    assert isinstance(dml_obj.all_coef, np.ndarray)
    assert isinstance(dml_obj.all_se, np.ndarray)
    assert isinstance(dml_obj.t_stat, np.ndarray)
    assert isinstance(dml_obj.pval, np.ndarray)

    # bootstrap
    assert dml_obj.boot_method == "normal"
    assert dml_obj.n_rep_boot == 500
    assert isinstance(dml_obj.boot_t_stat, np.ndarray)

    # sensitivity
    assert dml_obj.sensitivity_params is None
    assert isinstance(dml_obj.sensitivity_elements, dict)

    # fit method
    if isinstance(dml_obj, DoubleML):
        assert dml_obj.predictions is not None
        assert dml_obj.models is None

    # confint method
    assert dml_obj.confint().equals(dml_obj.confint(joint=False, level=0.95))

    # p_adjust method
    assert dml_obj.p_adjust().equals(dml_obj.p_adjust(method="romano-wolf"))
