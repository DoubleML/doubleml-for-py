.. _learners:

Machine learners, hyperparameters and hyperparameter tuning
-----------------------------------------------------------

The estimation of a double/debiased machine learning model involves the estimation of several nuisance function with
machine learning estimators.
Such machine learners are implemented in various Python and R packages.
The implementation of :ref:`DoubleML <doubleml_package>` is based on the meta-packages
`scikit-learn <https://scikit-learn.org/>`_ for Python and `mlr3 <https://mlr3.mlr-org.com/>`_ for R.
The interfaces to specify the learners, set hyperparameters and tune hyperparameters are described in the following
separately for :ref:`Python <learners_python>` and :ref:`R <learners_r>`

.. _learners_python:

Python: Machine learners and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Minimum requirements for learners
#################################

The minimum requirement for a learner to be used for nuisance models in the :ref:`DoubleML <doubleml_package>`
package is

    * The implementation of a ``fit()`` and ``predict()`` method.
      Some models, like :py:class:`doubleml.DoubleMLIRM` and :py:class:`doubleml.DoubleMLIIVM` require classifiers.
    * In case of classifiers, the learner needs to come with a ``predict_proba()`` instead of, or in addition to, a
      ``predict()`` method, see for example :py:meth:`sklearn.ensemble.RandomForestClassifier.predict_proba`.
    * In order to be able to use the ``set_ml_nuisance_params()`` method of :ref:`DoubleML <doubleml_package>` classes the
      learner additionally needs to come with a ``set_params()`` method,
      see for example :py:meth:`sklearn.ensemble.RandomForestRegressor.set_params`.
    * We further rely on the function :py:func:`sklearn.base.clone` which adds the requirement of a ``get_params()``
      method for a learner in order to be used for nuisance models of :ref:`DoubleML <doubleml_package>` model classes.

Most learners from `scikit-learn <https://scikit-learn.org/>`_ satisfy all these minimum requirements.

Specifying learners and set hyperparameters
###########################################

The learners are set during initialization of the :ref:`DoubleML <doubleml_package>` model classes
:py:class:`doubleml.DoubleMLPLR`, :py:class:`doubleml.DoubleMLPLIV`,
:py:class:`doubleml.DoubleMLIRM` and :py:class:`doubleml.DoubleMLIIVM`.
Lets simulate some data and consider the partially linear regression model.
We need to specify learners for the nuisance functions :math:`g_0(X) = E[Y|X]` and :math:`m_0(X) = E[D|X]`,
for example :py:class:`sklearn.ensemble.RandomForestRegressor`.

.. tabbed:: Python

    .. ipython:: python

        import doubleml as dml
        from doubleml.datasets import make_plr_CCDDHNR2018
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(1234)
        ml_g = RandomForestRegressor()
        ml_m = RandomForestRegressor()
        data = make_plr_CCDDHNR2018(return_type='DataFrame')
        obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m)
        dml_plr_obj.fit().summary

Setting hyperparameters:

    * We can also use pre-parametrized learners, like ``RandomForestRegressor(n_estimators=10)``.
    * Alternatively, hyperparameters can also be set after initialization via the method
      ``set_ml_nuisance_params(learner, treat_var, params)``


.. tabbed:: Python

    .. ipython:: python

        np.random.seed(1234)
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                      RandomForestRegressor(n_estimators=10),
                                      RandomForestRegressor())
        print(dml_plr_obj.fit().summary)

        np.random.seed(1234)
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                      RandomForestRegressor(),
                                      RandomForestRegressor())
        dml_plr_obj.set_ml_nuisance_params('ml_g', 'd', {'n_estimators': 10})
        print(dml_plr_obj.fit().summary)

Setting treatment-variable-specific or fold-specific hyperparameters:

    * In the multiple-treatment case, the method ``set_ml_nuisance_params(learner, treat_var, params)`` allows to set
      different hyperparameters for different treatment variables.
    * The method ``set_ml_nuisance_params(learner, treat_var, params)`` accepts dicts and lists for ``params``.
      A dict should be provided if for each fold the same hyperparameters should be used.
      Fold-specific parameters are supporter. To do so,  provide a nested list as ``params``, where the outer list is of
      length ``n_rep`` and the inner list of length ``n_folds``.


Hyperparameter tuning
#####################

Parameter tuning of learners for the nuisance functions of :ref:`DoubleML <doubleml_package>` models can be done via
the ``tune()`` method.
To illustrate the parameter tuning, we generate data from a sparse partially linear regression model.

.. tabbed:: Python

    .. ipython:: python

        import doubleml as dml
        import numpy as np

        np.random.seed(3141)
        n_obs = 200
        n_vars = 200
        theta = 3
        X = np.random.normal(size=(n_obs, n_vars))
        d = np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))
        y = theta * d + np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))
        dml_data = dml.DoubleMLData.from_arrays(X, y, d)

The hyperparameter-tuning is performed using either an exhaustive search over specified parameter values
implemented in :class:`sklearn.model_selection.GridSearchCV` or via a randomized search implemented in
:class:`sklearn.model_selection.RandomizedSearchCV`.

.. tabbed:: Python

    .. ipython:: python

        import doubleml as dml
        from sklearn.linear_model import Lasso

        ml_g = Lasso()
        ml_m = Lasso()
        dml_plr_obj = dml.DoubleMLPLR(dml_data, ml_g, ml_m)
        par_grids = {'ml_g': {'alpha': np.arange(0.05, 1., 0.1)},
                     'ml_m': {'alpha': np.arange(0.05, 1., 0.1)}}
        dml_plr_obj.tune(par_grids, search_mode='grid_search');
        print(dml_plr_obj.params)
        print(dml_plr_obj.fit().summary)

        np.random.seed(1234)
        par_grids = {'ml_g': {'alpha': np.arange(0.05, 1., 0.01)},
                     'ml_m': {'alpha': np.arange(0.05, 1., 0.01)}}
        dml_plr_obj.tune(par_grids, search_mode='randomized_search', n_iter_randomized_search=20);
        print(dml_plr_obj.params)
        print(dml_plr_obj.fit().summary)

Hyperparameter tuning can also be done with more sophisticated methods, like for example an iterative fitting along
a regularization path implemented in :py:class:`sklearn.linear_model.LassoCV`.
In this case the tuning should be done externally and the parameters can then be set via the
``set_ml_nuisance_params()`` method.

.. tabbed:: Python

    .. ipython:: python

        import doubleml as dml
        from sklearn.linear_model import LassoCV

        np.random.seed(1234)
        ml_g_tune = LassoCV().fit(dml_data.x, dml_data.y)
        ml_m_tune = LassoCV().fit(dml_data.x, dml_data.d)

        ml_g = Lasso()
        ml_m = Lasso()
        dml_plr_obj = dml.DoubleMLPLR(dml_data, ml_g, ml_m)
        dml_plr_obj.set_ml_nuisance_params('ml_g', 'd', {'alpha': ml_g_tune.alpha_});
        dml_plr_obj.set_ml_nuisance_params('ml_m', 'd', {'alpha': ml_m_tune.alpha_});
        print(dml_plr_obj.params)
        print(dml_plr_obj.fit().summary)


.. TODO: Also discuss other specification options like `tune_on_folds` or `scoring_methods`.

.. _learners_r:

R: Machine learners and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

