.. _data_backend:

The data-backend DoubleMLData
-----------------------------

:ref:`DoubleML <doubleml_package>` provides interfaces to dataframes as well as arrays. The usage of both interfaces is
demonstrated in the following. We download the Bonus data set from the Pennsylvania Reemployment Bonus experiment.

.. note::
    - In Python we use :py:class:`pandas.DataFrame` and :py:class:`numpy.ndarray`.
      The data can be fetched via :py:func:`doubleml.datasets.fetch_bonus`.
    - In R we use `data.table::data.table()`, `data.frame()`, and `matrix()`.
      The data can be fetched via `DoubleML::fetch_bonus()`.

.. tabbed:: Python

    .. ipython:: python

        from doubleml.datasets import fetch_bonus

        # Load data
        df_bonus = fetch_bonus('DataFrame')
        df_bonus.head(5)

.. tabbed:: R

    .. jupyter-execute::

        library(DoubleML)

        # Load data as data.table
        dt_bonus = fetch_bonus(return_type = "data.table")
        head(dt_bonus)

        # Load data as data.frame
        df_bonus = fetch_bonus(return_type = "data.frame")
        head(df_bonus)


DoubleMLData from dataframes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``DoubleMLData`` class serves as data-backend and can be initialized from a dataframe by
specifying the column ``y_col='inuidur1'`` serving as outcome variable :math:`Y`, the column(s) ``d_cols = 'tg'``
serving as treatment variable :math:`D` and the columns ``x_cols=['female', 'black', 'othrace', 'dep1', 'dep2', 'q2', 'q3', 'q4', 'q5', 'q6', 'agelt35', 'agegt54', 'durable', 'lusd', 'husd']``
specifying the confounders.

.. note::
    * In Python we use :py:class:`pandas.DataFrame`
      and the API reference can be found here :py:class:`doubleml.DoubleMLData`.
    * In R we use `data.table::data.table()` and the API reference can be found here `DoubleML::DoubleMLData`.
    * For initialization from the R base class `data.frame()` the API reference can be found here `DoubleML::data_ml_from_data_frame()`.

.. tabbed:: Python

    .. ipython:: python

        from doubleml import DoubleMLData

        # Specify the data and the variables for the causal model
        obj_dml_data_bonus = DoubleMLData(df_bonus,
                                          y_col='inuidur1',
                                          d_cols='tg',
                                          x_cols=['female', 'black', 'othrace', 'dep1', 'dep2',
                                                  'q2', 'q3', 'q4', 'q5', 'q6', 'agelt35', 'agegt54',
                                                  'durable', 'lusd', 'husd'],
                                          use_other_treat_as_covariate=True)
        print(obj_dml_data_bonus)

.. tabbed:: R

    .. jupyter-execute::

        # Specify the data and the variables for the causal model

        # From data.table object
        obj_dml_data_bonus = DoubleMLData$new(dt_bonus,
                                     y_col = "inuidur1",
                                     d_cols = "tg",
                                     x_cols = c("female", "black", "othrace", "dep1", "dep2",
                                                  "q2", "q3", "q4", "q5", "q6", "agelt35", "agegt54",
                                                  "durable", "lusd", "husd"),
                                     use_other_treat_as_covariate=TRUE)
        obj_dml_data_bonus

        # From dat.frame object
        obj_dml_data_bonus_df = double_ml_data_from_data_frame(df_bonus,
                                     y_col = "inuidur1",
                                     d_cols = "tg",
                                     x_cols = c("female", "black", "othrace", "dep1", "dep2",
                                                  "q2", "q3", "q4", "q5", "q6", "agelt35", "agegt54",
                                                  "durable", "lusd", "husd"),
                                     use_other_treat_as_covariate=TRUE)
        obj_dml_data_bonus_df

Comments on detailed specifications:

    * If ``x_cols`` is not specified, all variables (columns of the dataframe) which are neither specified as outcome
      variable ``y_col``, nor treatment variables ``d_cols``, nor instrumental variables ``z_cols`` are used as covariates.
    * In case of multiple treatment variables, the boolean ``use_other_treat_as_covariate`` indicates whether the other
      treatment variables should be added as covariates in each treatment-variable-specific learning task.
    * Instrumental variables for IV models have to be provided as ``z_cols``.

DoubleMLData from arrays and matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To introduce the array interface we generate a data set consisting of confounding variables ``X``, an outcome
variable ``y`` and a treatment variable ``d``

.. note::
    * In python we use :py:class:`numpy.ndarray`.
      and the API reference can be found here :py:func:`doubleml.DoubleMLData.from_arrays`.
    * In R we use the R base class `matrix()`
      and the API reference can be found here `DoubleML::double_ml_data_from_matrix()`.

.. tabbed:: Python

    .. ipython:: python

        import numpy as np

        # Generate data
        np.random.seed(3141)
        n_obs = 500
        n_vars = 100
        theta = 3
        X = np.random.normal(size=(n_obs, n_vars))
        d = np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))
        y = theta * d + np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))

.. tabbed:: R

    .. jupyter-execute::

        # Generate data
        set.seed(3141)
        n_obs = 500
        n_vars = 100
        theta = 3
        X = matrix(stats::rnorm(n_obs * n_vars), nrow = n_obs, ncol = n_vars)
        d = X[, 1:3, drop = FALSE] %*% c(5, 5, 5) + stats::rnorm(n_obs)
        y = theta * d + X[, 1:3, drop = FALSE] %*% c(5, 5, 5)  + stats::rnorm(n_obs)

To specify the data and the variables for the causal model from arrays we call

.. tabbed:: Python

    .. ipython:: python

        from doubleml import DoubleMLData

        obj_dml_data_sim = DoubleMLData.from_arrays(X, y, d)
        print(obj_dml_data_sim)

.. tabbed:: R

    .. jupyter-execute::

        obj_dml_data_sim = double_ml_data_from_matrix(X = X, y = y, d = d)
        obj_dml_data_sim