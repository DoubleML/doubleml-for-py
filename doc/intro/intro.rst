:parenttoc: True

Getting started
===============

The purpose of the following case-studies is to demonstrate the core functionalities of
:ref:`DoubleML <doubleml_package>`.

Data
----

For our case study we download the Bonus data set from the Pennsylvania Reemployment Bonus experiment and as a second
example we simulate data from a partially linear regression model.

.. tabbed:: Python

    .. ipython:: python

        import numpy as np
        from doubleml.datasets import fetch_bonus

        # Load bonus data
        df_bonus = fetch_bonus('DataFrame')
        print(df_bonus.head(5))

        # Simulate data
        np.random.seed(3141)
        n_obs = 500
        n_vars = 100
        theta = 3
        X = np.random.normal(size=(n_obs, n_vars))
        d = np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))
        y = theta * d + np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))

.. tabbed:: R

    .. jupyter-execute::

        library(DoubleML)

        # Load bonus data
        df_bonus = fetch_bonus(return_type="data.table")
        head(df_bonus)

        # Simulate data
        set.seed(3141)
        n_obs = 500
        n_vars = 100
        theta = 3
        X = matrix(rnorm(n_obs*n_vars), nrow=n_obs, ncol=n_vars)
        d = X[,1:3]%*%c(5,5,5) + rnorm(n_obs)
        y = theta*d + X[, 1:3]%*%c(5,5,5) + rnorm(n_obs)



The causal model
----------------

Exemplarily we specify a partially linear regression model (PLR). **Partially linear regression (PLR)** models take the
form

.. math::

    Y = D \theta_0 + g_0(X) + \zeta, & &\mathbb{E}(\zeta | D,X) = 0,

    D = m_0(X) + V, & &\mathbb{E}(V | X) = 0,

where :math:`Y` is the outcome variable and :math:`D` is the policy variable of interest.
The high-dimensional vector :math:`X = (X_1, \ldots, X_p)` consists of other confounding covariates,
and :math:`\zeta` and :math:`V` are stochastic errors.
For details about the implemented models in the :ref:`DoubleML <doubleml_package>` package we refer to the user guide
:ref:`models`.

The data-backend DoubleMLData
-----------------------------

:ref:`DoubleML <doubleml_package>` provides interfaces to dataframes as well as arrays.
Details on the data-backend and the interfaces can be found in the :ref:`user guide <data_backend>`.
The ``DoubleMLData`` class serves as data-backend and can be initialized from a dataframe by
specifying the column ``y_col='inuidur1'`` serving as outcome variable :math:`Y`, the column(s) ``d_cols = 'tg'``
serving as treatment variable :math:`D` and the columns ``x_cols`` specifying the confounders.
Alternatively an array interface can be used as shown below for the simulated data.

.. tabbed:: Python

    .. ipython:: python

        from doubleml import DoubleMLData

        # Specify the data and the variables for the causal model
        dml_data_bonus = DoubleMLData(df_bonus,
                                          y_col='inuidur1',
                                          d_cols='tg',
                                          x_cols=['female', 'black', 'othrace', 'dep1', 'dep2',
                                                  'q2', 'q3', 'q4', 'q5', 'q6', 'agelt35', 'agegt54',
                                                  'durable', 'lusd', 'husd'])
        print(dml_data_bonus)

        # array interface to DoubleMLData
        dml_data_sim = DoubleMLData.from_arrays(X, y, d)
        print(dml_data_sim)

.. tabbed:: R

    .. jupyter-execute::

        # Specify the data and variables for the causal model
        dml_data_bonus = DoubleMLData$new(df_bonus,
                                     y_col = "inuidur1",
                                     d_cols = "tg",
                                     x_cols = c("female", "black", "othrace", "dep1", "dep2",
                                                "q2", "q3", "q4", "q5", "q6", "agelt35", "agegt54",
                                                  "durable", "lusd", "husd"))
        print(dml_data_bonus)

        # matrix interface to DoubleMLData
        dml_data_sim = double_ml_data_from_matrix(X=X, y=y, d=d)
        dml_data_sim


Learners to estimate the nuisance models
------------------------------------------------

To estimate our partially linear regression (PLR) model with the double machine learning algorithm, we first have to
specify machine learners to estimate :math:`m_0` and :math:`g_0`. For the bonus data we use a random forest
regression model and for our simulated data from a sparse partially linear model we use a Lasso regression model.
The implementation of :ref:`DoubleML <doubleml_package>` is based on the meta-packages
`scikit-learn <https://scikit-learn.org/>`_ for Python and `mlr3 <https://mlr3.mlr-org.com/>`_ for R (Lang et al, 2019).
For details on the specification of learners and their hyperparameters we refer to the user guide :ref:`learners`.

.. tabbed:: Python

    .. ipython:: python

        from sklearn.base import clone
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LassoCV

        learner = RandomForestRegressor(n_estimators = 500, max_features = 'sqrt', max_depth= 5)
        ml_g_bonus = clone(learner)
        ml_m_bonus = clone(learner)

        learner = LassoCV()
        ml_g_sim = clone(learner)
        ml_m_sim = clone(learner)

.. tabbed:: R

    .. jupyter-execute::

        library(mlr3)
        library(mlr3learners)
        # surpress messages from mlr3 package during fitting
        lgr::get_logger("mlr3")$set_threshold("warn")

        learner = lrn("regr.ranger", num.trees=500, mtry=floor(sqrt(n_vars)), max.depth=5, min.node.size=2)
        ml_g_bonus = learner$clone()
        ml_m_bonus = learner$clone()

        learner = lrn("regr.cv_glmnet", s="lambda.min")
        ml_g_sim = learner$clone()
        ml_m_sim = learner$clone()


Cross-fitting, DML algorithms and Neyman-orthogonal score functions
-------------------------------------------------------------------

When initializing the object for PLR models ``DoubleMLPLR``, we can further set parameters specifying the
resampling: The number of folds used for cross-fitting ``n_folds`` (defaults to ``n_folds = 5``) as well as the number
of repetitions when applying repeated cross-fitting ``n_rep`` (defaults to ``n_rep = 1``).
Additionally, one can choose between the algorithms ``'dml1'`` and  ``'dml2'`` via ``dml_procedure`` (defaults to
``'dml2'``).
Depending on the causal model, one can further choose between different Neyman-orthogonal score / moment functions.
For the PLR model the default ``score`` is ``'partialling out'``.

The user guide provides details about the :ref:`resampling`, the :ref:`algorithms`
and the :ref:`scores`.

Estimate double/debiased machine learning models
------------------------------------------------

We now initialize ``DoubleMLPLR`` objects for our examples using default parameters.
The models are estimated by calling the ``fit()`` method and we can for example inspect the estimated treatment effect
using the ``summary`` property.
A more detailed result summary can be obtained via the string-representation of the object.
Besides the ``fit()`` method :ref:`DoubleML <doubleml_package>` model classes also provide functionalities to perform
statistical inference like ``bootstrap()``, ``confint()`` and ``p_adjust()``, for details see the user guide
:ref:`se_confint`.

.. tabbed:: Python

    .. ipython:: python

        from doubleml import DoubleMLPLR
        np.random.seed(3141)
        obj_dml_plr_bonus = DoubleMLPLR(dml_data_bonus, ml_g_bonus, ml_m_bonus)
        obj_dml_plr_bonus.fit();
        print(obj_dml_plr_bonus)

        obj_dml_plr_sim = DoubleMLPLR(dml_data_sim, ml_g_sim, ml_m_sim)
        obj_dml_plr_sim.fit();
        print(obj_dml_plr_sim)

.. tabbed:: R

    .. jupyter-execute::

        set.seed(3141)
        obj_dml_plr_bonus = DoubleMLPLR$new(dml_data_bonus, ml_g=ml_g_bonus, ml_m=ml_m_bonus)
        obj_dml_plr_bonus$fit()
        print(obj_dml_plr_bonus)

        obj_dml_plr_sim = DoubleMLPLR$new(dml_data_sim, ml_g=ml_g_sim, ml_m=ml_m_sim)
        obj_dml_plr_sim$fit()
        print(obj_dml_plr_sim)

References
++++++++++

* Lang, M., Binder, M., Richter, J., Schratz, P., Pfisterer, F., Coors, S., Au, Q., Casalicchio, G., Kotthoff, L., Bischl, B. (2019), mlr3: A modern object-oriented machine learing framework in R. Journal of Open Source Software, `doi:10.21105/joss.01903 <(doi:10.21105/joss.01903)[10.21105/joss.01903]>`_.


