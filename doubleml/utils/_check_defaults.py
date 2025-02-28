def _check_basic_defaults_before_fit(dml_obj):
    assert dml_obj.n_rep_boot is None
    assert dml_obj.boot_method is None
    assert dml_obj.framework is None
    assert dml_obj.sensitivity_params is None
    assert dml_obj.boot_t_stat is None
    assert dml_obj._draw_sample_splitting


def _fit_bootstrap(dml_obj):
    dml_obj.fit()
    dml_obj.bootstrap()


def _check_basic_defaults_after_fit(dml_obj):
    assert dml_obj.n_folds == 5
    assert dml_obj.n_rep == 1

    # fit method
    assert dml_obj.predictions is not None
    assert dml_obj.models is None

    # bootstrap method
    assert dml_obj.boot_method == "normal"
    assert dml_obj.n_rep_boot == 500

    # confint method
    assert dml_obj.confint().equals(dml_obj.confint(joint=False, level=0.95))

    # p_adjust method
    assert dml_obj.p_adjust().equals(dml_obj.p_adjust(method="romano-wolf"))
