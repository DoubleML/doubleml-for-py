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

        import numpy as np
        import doubleml as dml
        from doubleml.datasets import make_plr_CCDDHNR2018
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.base import clone

        learner = RandomForestRegressor(max_depth=2, n_estimators=10)
        ml_g = clone(learner)
        ml_m = clone(learner)
        np.random.seed(1111)
        data = make_plr_CCDDHNR2018(alpha=0.5, n_obs=500, dim_x=20, return_type='DataFrame')
        obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m)
        print(dml_plr_obj.fit())

.. tabbed:: R

    .. jupyter-execute::

        library(DoubleML)
        library(mlr3)
        library(mlr3learners)
        library(data.table)
        lgr::get_logger("mlr3")$set_threshold("warn")

        learner = lrn("regr.ranger", num.trees = 10, max.depth = 2)
        ml_g = learner$clone()
        ml_m = learner$clone()
        set.seed(1111)
        data = make_plr_CCDDHNR2018(alpha=0.5, n_obs=500, dim_x=20, return_type='data.table')
        obj_dml_data = DoubleMLData$new(data, y_col="y", d_cols="d")
        dml_plr_obj = DoubleMLPLR$new(obj_dml_data, ml_g, ml_m)
        dml_plr_obj$fit()
        print(dml_plr_obj)


Partially linear IV regression model (PLIV)
+++++++++++++++++++++++++++++++++++++++++++

.. include:: ../shared/models/pliv.rst

.. include:: ../shared/causal_graphs/pliv_iivm_causal_graph.rst

``DoubleMLPLIV`` implements PLIV models.
Estimation is conducted via its ``fit()`` method:

.. tabbed:: Python

    .. ipython:: python
        :okwarning:

        import numpy as np
        import doubleml as dml
        from doubleml.datasets import make_pliv_CHS2015
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.base import clone

        learner = RandomForestRegressor(max_depth=2, n_estimators=10)
        ml_g = clone(learner)
        ml_m = clone(learner)
        ml_r = clone(learner)
        np.random.seed(2222)
        data = make_pliv_CHS2015(alpha=0.5, n_obs=500, dim_x=20, dim_z=1, return_type='DataFrame')
        obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_cols='Z1')
        dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data, ml_g, ml_m, ml_r)
        print(dml_pliv_obj.fit())

.. tabbed:: R

    .. jupyter-execute::

        library(DoubleML)
        library(mlr3)
        library(mlr3learners)
        library(data.table)

        learner = lrn("regr.ranger", num.trees = 10, max.depth = 2)
        ml_g = learner$clone()
        ml_m = learner$clone()
        ml_r = learner$clone()
        set.seed(2222)
        data = make_pliv_CHS2015(alpha=0.5, n_obs=500, dim_x=20, dim_z=1, return_type="data.table")
        obj_dml_data = DoubleMLData$new(data, y_col="y", d_col = "d", z_cols= "Z1")
        dml_pliv_obj = DoubleMLPLIV$new(obj_dml_data, ml_g, ml_m, ml_r)
        dml_pliv_obj$fit()
        print(dml_pliv_obj)


Interactive regression model (IRM)
++++++++++++++++++++++++++++++++++

.. include:: ../shared/models/irm.rst

.. include:: ../shared/causal_graphs/plr_irm_causal_graph.rst

``DoubleMLIRM`` implements IRM models.
Estimation is conducted via its ``fit()`` method:

.. tabbed:: Python

    .. ipython:: python

        import numpy as np
        import doubleml as dml
        from doubleml.datasets import make_irm_data
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        ml_g = RandomForestRegressor(max_depth=2, n_estimators=10)
        ml_m = RandomForestClassifier(max_depth=2, n_estimators=10)
        np.random.seed(3333)
        data = make_irm_data(theta=0.5, n_obs=500, dim_x=20, return_type='DataFrame')
        obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
        dml_irm_obj = dml.DoubleMLIRM(obj_dml_data, ml_g, ml_m)
        print(dml_irm_obj.fit())

.. tabbed:: R

    .. jupyter-execute::

        library(DoubleML)
        library(mlr3)
        library(mlr3learners)
        library(data.table)

        set.seed(3333)
        ml_g = lrn("regr.ranger", num.trees = 10, max.depth = 2)
        ml_m = lrn("classif.ranger", num.trees = 10, max.depth = 2)
        data = make_irm_data(theta=0.5, n_obs=500, dim_x=20, return_type="data.table")
        obj_dml_data = DoubleMLData$new(data, y_col="y", d_cols="d")
        dml_irm_obj = DoubleMLIRM$new(obj_dml_data, ml_g, ml_m)
        dml_irm_obj$fit()
        print(dml_irm_obj)



Interactive IV model (IIVM)
+++++++++++++++++++++++++++

.. include:: ../shared/models/iivm.rst

.. include:: ../shared/causal_graphs/pliv_iivm_causal_graph.rst

``DoubleMLIIVM`` implements IIVM models.
Estimation is conducted via its ``fit()`` method:

.. tabbed:: Python

    .. ipython:: python
        :okwarning:

        import numpy as np
        import doubleml as dml
        from doubleml.datasets import make_iivm_data
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        ml_g = RandomForestRegressor(max_depth=2, n_estimators=10)
        ml_m = RandomForestClassifier(max_depth=2, n_estimators=10)
        ml_r = RandomForestClassifier(max_depth=2, n_estimators=10)
        np.random.seed(4444)
        data = make_iivm_data(theta=1., n_obs=500, dim_x=20, return_type='DataFrame')
        obj_dml_data = dml.DoubleMLData(data, 'y', 'd', z_cols='z')
        dml_iivm_obj = dml.DoubleMLIIVM(obj_dml_data, ml_g, ml_m, ml_r)
        print(dml_iivm_obj.fit())

.. tabbed:: R

    .. jupyter-execute::

        library(DoubleML)
        library(mlr3)
        library(mlr3learners)
        library(data.table)

        set.seed(4444)
        ml_g = lrn("regr.ranger", num.trees = 10, max.depth = 2)
        ml_m = lrn("classif.ranger", num.trees = 10, max.depth = 2)
        ml_r = ml_m$clone()
        data = make_iivm_data(theta=1, n_obs=500, dim_x=20, return_type="data.table")
        obj_dml_data = DoubleMLData$new(data, y_col="y", d_cols="d", z_cols="z")
        dml_iivm_obj = DoubleMLIIVM$new(obj_dml_data, ml_g, ml_m, ml_r)
        dml_iivm_obj$fit()
        print(dml_iivm_obj)


