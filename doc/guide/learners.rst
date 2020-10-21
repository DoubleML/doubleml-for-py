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



Hyperparameter tuning
#####################



.. _learners_r:

R: Machine learners and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

