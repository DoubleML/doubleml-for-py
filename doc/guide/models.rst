Models
----------

Partially linear regression model (PLR)
+++++++++++++++++++++++++++++++++++++++

.. include:: ../shared/models/plr.rst

.. include:: ../shared/causal_graphs/plr_irm_causal_graph.rst

:class:`~doubleml.double_ml_plr.DoubleMLPLR` implements PLR models.
Estimation is conducted via its :meth:`~doubleml.double_ml_plr.DoubleMLPLR.fit` method:

.. tabs::

    .. code-tab:: py

        >>> import doubleml as dml
        >>> from doubleml.datasets import make_plr_data
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from sklearn.base import clone

        >>> learner = RandomForestRegressor(max_depth=2, n_estimators=10)
        >>> ml_g = clone(learner)
        >>> ml_m = clone(learner)
        >>> data = make_plr_data()
        >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
        >>> dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m)
        >>> dml_plr_obj.fit()
        >>> dml_plr_obj.summary
               coef   std err         t     P>|t|     2.5 %    97.5 %
        d  0.515147  0.105472  4.884214  0.000001  0.308426  0.721868

    .. code-tab:: r R

        > # R-code here
        > a=5


Partially linear IV regression model (PLIV)
+++++++++++++++++++++++++++++++++++++++++++

.. include:: ../shared/models/pliv.rst

.. include:: ../shared/causal_graphs/pliv_iivm_causal_graph.rst

:class:`~doubleml.double_ml_pliv.DoubleMLPLIV` implements PLIV models.
Estimation is conducted via its :meth:`~doubleml.double_ml_pliv.DoubleMLPLIV.fit` method:

.. tabs::

    .. code-tab:: py

        >>> import doubleml as dml
        >>> from doubleml.datasets import make_pliv_data
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from sklearn.base import clone

        >>> learner = RandomForestRegressor(max_depth=2, n_estimators=10)
        >>> ml_g = clone(learner)
        >>> ml_m = clone(learner)
        >>> ml_r = clone(learner)
        >>> data = make_pliv_data()
        >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_col='z')
        >>> dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data, ml_g, ml_m, ml_r)
        >>> dml_pliv_obj.fit()
        >>> dml_pliv_obj.summary
               coef   std err         t     P>|t|     2.5 %    97.5 %
        d  1.310476  1.172527  1.117651  0.263716 -0.987635  3.608586

    .. code-tab:: r R

        > # R-code here
        > a=5


Interactive regression model (IRM)
++++++++++++++++++++++++++++++++++

.. include:: ../shared/models/irm.rst

.. include:: ../shared/causal_graphs/plr_irm_causal_graph.rst

:class:`~doubleml.double_ml_irm.DoubleMLIRM` implements IRM models.
Estimation is conducted via its :meth:`~doubleml.double_ml_irm.DoubleMLIRM.fit` method:

.. tabs::

    .. code-tab:: py

        >>> import doubleml as dml
        >>> from doubleml.datasets import make_irm_data
        >>> from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        >>> ml_g = RandomForestRegressor(max_depth=2, n_estimators=10)
        >>> ml_m = RandomForestClassifier(max_depth=2, n_estimators=10)
        >>> data = make_irm_data()
        >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
        >>> dml_irm_obj = dml.DoubleMLIRM(obj_dml_data, ml_g, ml_m)
        >>> dml_irm_obj.fit()
        >>> dml_irm_obj.summary
               coef   std err         t     P>|t|    2.5 %    97.5 %
        d  0.624459  0.238203  2.621541  0.008753  0.15759  1.091329

    .. code-tab:: r R

        > # R-code here
        > a=5


Interactive IV model (IIVM)
+++++++++++++++++++++++++++

.. include:: ../shared/models/iivm.rst

.. include:: ../shared/causal_graphs/pliv_iivm_causal_graph.rst

:class:`~doubleml.double_ml_iivm.DoubleMLIIVM` implements IIVM models.
Estimation is conducted via its :meth:`~doubleml.double_ml_iivm.DoubleMLIIVM.fit` method:

.. tabs::

    .. code-tab:: py

        >>> import doubleml as dml
        >>> from doubleml.datasets import make_iivm_data
        >>> from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        >>> ml_g = RandomForestRegressor(max_depth=2, n_estimators=10)
        >>> ml_m = RandomForestClassifier(max_depth=2, n_estimators=10)
        >>> ml_r = RandomForestClassifier(max_depth=2, n_estimators=10)
        >>> data = make_iivm_data()
        >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_col='z')
        >>> dml_iivm_obj = dml.DoubleMLIIVM(obj_dml_data, ml_g, ml_m, ml_r)
        >>> dml_iivm_obj.fit()
        >>> dml_iivm_obj.summary
               coef   std err         t     P>|t|      2.5 %     97.5 %
        d  2.190418  8.223016  0.266376  0.789949 -13.926397  18.307233

    .. code-tab:: r R

        > # R-code here
        > a=5

