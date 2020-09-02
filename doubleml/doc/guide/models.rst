Key models
----------

Partially linear regression model (PLR)
+++++++++++++++++++++++++++++++++++++++

.. currentmodule:: doubleml.double_ml_plr

.. include:: ../shared/models/plr.rst

:class:`DoubleMLPLR` implements PLR models. Estimation is conducted via its :meth:`DoubleMLPLR.fit` method:

.. ipython:: python

    import doubleml as dml
    from doubleml.datasets import make_plr_data
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.base import clone

    learner = RandomForestRegressor(max_depth=2, n_estimators=10)
    ml_learners = {'ml_m': clone(learner), 'ml_g': clone(learner)}
    data = make_plr_data()
    obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_learners)
    dml_plr_obj.fit()
    dml_plr_obj.summary


Partially linear IV regression model (PLIV)
+++++++++++++++++++++++++++++++++++++++++++

.. currentmodule:: doubleml.double_ml_pliv

.. include:: ../shared/models/pliv.rst

:class:`DoubleMLPLIV` implements PLIV models. Estimation is conducted via its :meth:`DoubleMLPLIV.fit` method:

.. ipython:: python

    import doubleml as dml
    from doubleml.datasets import make_pliv_data
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.base import clone

    learner = RandomForestRegressor(max_depth=2, n_estimators=10)
    ml_learners = {'ml_m': clone(learner), 'ml_g': clone(learner), 'ml_r': clone(learner)}
    data = make_pliv_data()
    obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_col='z')
    dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data, ml_learners)
    dml_pliv_obj.fit()
    dml_pliv_obj.summary


Interactive regression model (IRM)
++++++++++++++++++++++++++++++++++

.. currentmodule:: doubleml.double_ml_irm

.. include:: ../shared/models/irm.rst

:class:`DoubleMLIRM` implements IRM models. Estimation is conducted via its :meth:`DoubleMLIRM.fit` method:

.. ipython:: python

    import doubleml as dml
    from doubleml.datasets import make_irm_data
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    ml_learners = {'ml_m': RandomForestClassifier(max_depth=2, n_estimators=10),
                   'ml_g': RandomForestRegressor(max_depth=2, n_estimators=10)}
    data = make_irm_data()
    obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
    dml_irm_obj = dml.DoubleMLIRM(obj_dml_data, ml_learners)
    dml_irm_obj.fit()
    dml_irm_obj.summary


Interactive IV model (IIVM)
+++++++++++++++++++++++++++++++++++++

.. currentmodule:: doubleml.double_ml_iivm

.. include:: ../shared/models/iivm.rst

:class:`DoubleMLIIVM` implements IIVM models. Estimation is conducted via its :meth:`DoubleMLIIVM.fit` method:

.. ipython:: python

    import doubleml as dml
    from doubleml.datasets import make_iivm_data
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    ml_learners = {'ml_m': RandomForestClassifier(max_depth=2, n_estimators=10),
                   'ml_g': RandomForestRegressor(max_depth=2, n_estimators=10),
                   'ml_r': RandomForestClassifier(max_depth=2, n_estimators=10)}
    data = make_iivm_data()
    obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_col='z')
    dml_iivm_obj = dml.DoubleMLIIVM(obj_dml_data, ml_learners)
    dml_iivm_obj.fit()
    dml_iivm_obj.summary

