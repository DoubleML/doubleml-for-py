:parenttoc: True

Getting started
===============

The purpose of the following case-studies is to demonstrate the core functionalities of ``doubleml``.

Data and the causal model
-------------------------

:ref:`DoubleML <doubleml_package>` provides interfaces to dataframes as well as arrays. The usage of both interfaces is
demonstrated in the following. We download the 401(k) data set.

.. note::
    * In python we use :py:class:`pandas.DataFrame` and :py:class:`numpy.ndarray`.
    * In R we use

.. tabbed:: Python

    .. ipython:: python

        from doubleml.datasets import fetch_401K

        # Load data
        df_401k = fetch_401K('DataFrame')
        df_401k.head(5)

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)

Partially linear regression (PLR) models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Exemplarily we specify a partially linear regression model (PLR). **Partially linear regression (PLR)** models take the
form

.. math::

    Y = D \theta_0 + g_0(X) + \zeta, & &\mathbb{E}(\zeta | D,X) = 0,

    D = m_0(X) + V, & &\mathbb{E}(V | X) = 0,

where :math:`Y` is the outcome variable and :math:`D` is the policy variable of interest.
The high-dimensional vector :math:`X = (X_1, \ldots, X_p)` consists of other confounding covariates,
and :math:`\zeta` and :math:`V` are stochastic errors.

DoubleMLData from dataframes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``DoubleMLData`` class serves as data-backend and can be initialized from a dataframe by
specifying the column ``y_col='net_tfa'`` serving as outcome variable :math:`Y`, the column(s) ``d_cols = 'e401'``
serving as treatment variable :math:`D` and the columns ``x_cols=['age', 'inc', 'educ', 'fsize', 'marr', 'twoearn', 'db', 'pira', 'hown']``
specifying the confounders.

.. note::
    * In python we use :py:class:`pandas.DataFrame`.
    * In R we use

.. tabbed:: Python

    .. ipython:: python

        from doubleml import DoubleMLData

        # Specify the data and the variables for the causal model
        obj_dml_data_401k = DoubleMLData(df_401k,
                                         y_col='net_tfa',
                                         d_cols='e401',
                                         x_cols=['age', 'inc', 'educ', 'fsize', 'marr',
                                                 'twoearn', 'db', 'pira', 'hown'])
        print(obj_dml_data_401k)

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)


DoubleMLData from arrays
^^^^^^^^^^^^^^^^^^^^^^^^

To introduce the array interface we generate a data set consisting of confounding variables ``X``, an outcome
variable ``y`` and a treatment variable ``d``

.. note::
    * In python we use :py:class:`numpy.ndarray`.
    * In R we use

.. tabbed:: Python

    .. ipython:: python

        import numpy as np

        # Generate data
        n_obs = 500
        n_vars = 100
        theta = 3
        X = np.random.normal(size=(n_obs, n_vars))
        d = np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))
        y = theta * d + np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)

To specify the data and the variables for the causal model from arrays we call

.. tabbed:: Python

    .. ipython:: python

        from doubleml import DoubleMLData

        obj_dml_data_sim = DoubleMLData.from_arrays(X, y, d)
        print(obj_dml_data_sim)

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)

Estimate a causal model with double/debiased machine learning
-------------------------------------------------------------

Machine learners to estimate the nuisance models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To estimate our partially linear regression (PLR) model with the double machine learning algorithm, we first have to
specify machine learners to estimate :math:`m_0` and :math:`g_0`. For the 401(k) data we use
a random forest regression model
and for our simulated data from a sparse linear model we use a Lasso regression model.

.. note::
    * In python the machine learners are implemented in :py:class:`~sklearn.ensemble.RandomForestRegressor` from :py:mod:`sklearn.ensemble` and :py:class:`~sklearn.linear_model.Lasso` from :py:mod:`sklearn.linear_model`.
    * In R we use

.. tabbed:: Python

    .. ipython:: python

        from sklearn.base import clone
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Lasso

        learner = RandomForestRegressor(max_depth=2, n_estimators=100)
        ml_g_401k = clone(learner)
        ml_m_401k = clone(learner)

        learner = Lasso(alpha=np.sqrt(np.log(n_vars)/(n_obs)))
        ml_g_sim = clone(learner)
        ml_m_sim = clone(learner)

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)

Cross-fitting, DML algorithms and Neyman-orthogonal score functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When initializing the object for PLR models ``DoubleMLPLR``, we can further set parameters specifying the
resampling: The number of folds used for cross-fitting ``n_folds`` (defaults to ``n_folds = 5``) as well as the number
of repetitions when applying repeated cross-fitting ``n_rep`` (defaults to ``n_rep = 1``).
Additionally, one can choose between the algorithms ``'dml1'`` and  ``'dml2'`` via ``dml_procedure``. Depending on the
causal model, one can further choose between different Neyman-orthogonal score / moment functions.

DoubleMLPLR: Double/debiased machine learning for partially linear regression models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We now initialize ``DoubleMLPLR`` objects for our examples using default parameters


.. tabbed:: Python

    .. ipython:: python

        from doubleml import DoubleMLPLR
        obj_dml_plr_401k = DoubleMLPLR(obj_dml_data_401k, ml_g_401k, ml_m_401k)
        obj_dml_plr_sim = DoubleMLPLR(obj_dml_data_sim, ml_g_sim, ml_m_sim)

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)

Estimate double/debiased machine learning models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The models are estimated by calling the ``fit()`` method and we can inspect the estimated treatment effect using the
``summary`` property.


.. tabbed:: Python

    .. ipython:: python

        obj_dml_plr_401k.fit()
        print(obj_dml_plr_401k.summary)

        obj_dml_plr_sim.fit()
        print(obj_dml_plr_sim.summary)

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)
