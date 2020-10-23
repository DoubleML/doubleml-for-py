.. _models:

Models
----------

Partially linear regression model (PLR)
+++++++++++++++++++++++++++++++++++++++

.. include:: ../shared/models/plr.rst

.. include:: ../shared/causal_graphs/plr_irm_causal_graph.rst

``DoubleMLPLR`` implements PLR models.
Estimation is conducted via its ``fit()`` method:

.. tabbed:: Python

    .. ipython:: python

        import doubleml as dml
        from doubleml.datasets import make_plr_CCDDHNR2018
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.base import clone

        learner = RandomForestRegressor(max_depth=2, n_estimators=10)
        ml_g = clone(learner)
        ml_m = clone(learner)
        data = make_plr_CCDDHNR2018(return_type='DataFrame')
        obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m)
        print(dml_plr_obj.fit())

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)


Partially linear IV regression model (PLIV)
+++++++++++++++++++++++++++++++++++++++++++

.. include:: ../shared/models/pliv.rst

.. include:: ../shared/causal_graphs/pliv_iivm_causal_graph.rst

``DoubleMLPLIV`` implements PLIV models.
Estimation is conducted via its ``fit()`` method:

.. tabbed:: Python

    .. ipython:: python
        :okwarning:

        import doubleml as dml
        from doubleml.datasets import make_pliv_data
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.base import clone

        learner = RandomForestRegressor(max_depth=2, n_estimators=10)
        ml_g = clone(learner)
        ml_m = clone(learner)
        ml_r = clone(learner)
        data = make_pliv_data(return_type='DataFrame')
        obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_cols='z')
        dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data, ml_g, ml_m, ml_r)
        print(dml_pliv_obj.fit())

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)


Interactive regression model (IRM)
++++++++++++++++++++++++++++++++++

.. include:: ../shared/models/irm.rst

.. include:: ../shared/causal_graphs/plr_irm_causal_graph.rst

``DoubleMLIRM`` implements IRM models.
Estimation is conducted via its ``fit()`` method:

.. tabbed:: Python

    .. ipython:: python

        import doubleml as dml
        from doubleml.datasets import make_irm_data
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        ml_g = RandomForestRegressor(max_depth=2, n_estimators=10)
        ml_m = RandomForestClassifier(max_depth=2, n_estimators=10)
        data = make_irm_data(return_type='DataFrame')
        obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
        dml_irm_obj = dml.DoubleMLIRM(obj_dml_data, ml_g, ml_m)
        print(dml_irm_obj.fit())

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)


Interactive IV model (IIVM)
+++++++++++++++++++++++++++

.. include:: ../shared/models/iivm.rst

.. include:: ../shared/causal_graphs/pliv_iivm_causal_graph.rst

``DoubleMLIIVM`` implements IIVM models.
Estimation is conducted via its ``fit()`` method:

.. tabbed:: Python

    .. ipython:: python
        :okwarning:

        import doubleml as dml
        from doubleml.datasets import make_iivm_data
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        ml_g = RandomForestRegressor(max_depth=2, n_estimators=10)
        ml_m = RandomForestClassifier(max_depth=2, n_estimators=10)
        ml_r = RandomForestClassifier(max_depth=2, n_estimators=10)
        data = make_iivm_data(return_type='DataFrame')
        obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_cols='z')
        dml_iivm_obj = dml.DoubleMLIIVM(obj_dml_data, ml_g, ml_m, ml_r)
        print(dml_iivm_obj.fit())

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)

