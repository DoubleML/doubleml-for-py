Getting started
===============

The purpose of the following case-study is to demonstrate the core functionalities of ``DoubleML``. ``DoubleML``
provides interfaces to ``pandas`` DataFrames as well as ``numpy`` arrays. The usage of both interfaces is demonstrated
with two case studies.

Example 1: 401(k) data set
--------------------------

The following real data application is taken from Chernozhukov et al. (2017) illustrates the usage of ``DoubleMLPLR``
for carrying out inference on the main causal parameter in a partially linear regression model (PLR). **Partially linear
regression (PLR)** models take the form

.. math::

    Y = D \theta_0 + g_0(X) + \zeta, & &\mathbb{E}(\zeta | D,X) = 0,

    D = m_0(X) + V, & &\mathbb{E}(V | X) = 0,

where :math:`Y` is the outcome variable and :math:`D` is the policy variable of interest.
The high-dimensional vector :math:`X = (X_1, \ldots, X_p)` consists of other confounding covariates,
and :math:`\zeta` and :math:`V` are stochastic errors.

.. ipython:: python

    from doubleml import DoubleMLData, DoubleMLPLR
    from doubleml.datasets import fetch_401K

    from sklearn.base import clone
    from sklearn.ensemble import RandomForestRegressor

    # Load data
    data = fetch_401K()

    # Specify the data and the variables for the causal model
    dml_data_obj = DoubleMLData(data,
                                y_col='net_tfa',
                                d_cols='e401',
                                x_cols=['age', 'inc', 'educ', 'fsize', 'marr', 'twoearn', 'db', 'pira', 'hown'])

    # Specify the learners for the nuisance functions and initialize the DoubleMLPLR object
    learner = RandomForestRegressor(max_depth=2, n_estimators=100)
    ml_learners = {'ml_m': clone(learner),
                   'ml_g': clone(learner)}
    dml_plr_obj = DoubleMLPLR(dml_data_obj, ml_learners)

    # Fit the PLR model with double machine learning
    dml_plr_obj.fit()

    # Inspect the results
    dml_plr_obj.summary


Example 2: Simulated data
-------------------------
.. ipython:: python

    from doubleml import DoubleMLData, DoubleMLPLR
    from doubleml.datasets import fetch_401K

    import numpy as np
    from sklearn.base import clone
    from sklearn.linear_model import Lasso

    # Generate data
    n_obs = 500
    n_vars = 100
    n_nonzero = 3
    beta_d = np.random.uniform(0, 5, size=n_nonzero)
    beta_y = np.random.uniform(0, 5, size=n_nonzero)
    theta = 3
    X = np.random.normal(size=(n_obs, n_vars))
    d = np.dot(X[:, :n_nonzero], beta_d) + np.random.standard_normal(size=(n_obs,))
    y = theta * d + np.dot(X[:, :n_nonzero], beta_y) + np.random.standard_normal(size=(n_obs,))

    # Specify the data and the variables for the causal model
    dml_data_obj = DoubleMLData.from_arrays(X, y, d)

    # Specify the learners for the nuisance functions and initialize the DoubleMLPLR object
    learner = Lasso(alpha=np.sqrt(np.log(n_vars)/(n_obs)))
    ml_learners = {'ml_m': clone(learner),
                   'ml_g': clone(learner)}
    dml_plr_obj = DoubleMLPLR(dml_data_obj, ml_learners)

    # Fit the PLR model with double machine learning
    dml_plr_obj.fit()

    # Inspect the results
    dml_plr_obj.summary
