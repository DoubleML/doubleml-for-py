Machine learners, hyperparameters and hyperparameter tuning
-----------------------------------------------------------

The estimation of a double/debiased machine learning model involves the estimation of several nuisance function with
machine learning estimators.
Such machine learners are implemented in various Python and R packages.
The implementation of :ref:`DoubleML <doubleml_package>` is based on the meta-packages
`scikit-learn <https://scikit-learn.org/>`_ for Python and `mlr3 <https://mlr3.mlr-org.com/>`_.
The interfaces to specify the learners, set hyperparameters and tune hyperparameters are described in the following
separately for :ref:`Python <learners_python>` and :ref:`R <learners_r>`

.. _learners_python:

Python: Machine learners and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Minimum requirements for learners
#################################

The minimum requirement for a learner to be used for a nuisance model in the :ref:`DoubleML <doubleml_package>` is

    * The implementation of a ``fit()`` and ``predict()`` method.
      Some models, like :py:class:`doubleml.DoubleMLIRM` and :py:class:`doubleml.DoubleMLIIVM` require classifiers.
    * In case of classifiers, the learner needs to come with a ``predict_proba()`` instead of or in addition to a
      ``predict()`` method, see for example :py:meth:`sklearn.ensemble.RandomForestClassifier.predict_proba`.
    * In order to be able to use the ``set_ml_nuisance_params()`` method of :ref:`DoubleML <doubleml_package>` classes the
      learner additional needs to come with a ``set_params()`` method,
      see for example :py:meth:`sklearn.ensemble.RandomForestRegressor.set_params`.

Most learners from `scikit-learn <https://scikit-learn.org/>`_ satisfy all these minimum requirements.

Specifying learners and set hyperparameters
###########################################

The learners are set during initialization of the :ref:`DoubleML <doubleml_package>` model classes
:py:class:`doubleml.DoubleMLPLR`, :py:class:`doubleml.DoubleMLPLIV`,
:py:class:`doubleml.DoubleMLIRM` and :py:class:`doubleml.DoubleMLIIVM`.
Lets simulate some data and consider the partial linear regression model.
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

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)

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
        dml_plr_obj.fit()
        print(dml_plr_obj.summary)

        np.random.seed(1234)
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data,
                                      RandomForestRegressor(),
                                      RandomForestRegressor())
        dml_plr_obj.set_ml_nuisance_params('ml_g', 'd', {'n_estimators': 10})
        dml_plr_obj.fit()
        print(dml_plr_obj.summary)

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)

Setting treatment-variable-specific or fold-specific hyperparameters:

    * In the multiple treatment case, the method ``set_ml_nuisance_params(learner, treat_var, params)`` allows to set
      different hyperparameters for different treatment variables.
    * The method ``set_ml_nuisance_params(learner, treat_var, params)`` accepts dicts and lists for ``params``.
      A dict should be provided if for each fold the same hyperparameters should be used.
      Fold-specific parameters are supporter. To do so,  provide a nested list as ``params``, where the outer list is of
      length n_rep and the inner list of length n_folds.


Hyperparameter tuning
#####################



.. _learners_r:

R: Machine learners and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

