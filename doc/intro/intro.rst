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

.. tabs::

    .. code-tab:: py

        >>> from doubleml.datasets import fetch_401K

        >>> # Load data
        >>> df_401k = fetch_401K()
        >>> df_401k.head(5)
              nifa  net_tfa        tw  age      inc  ...  twoearn  e401  p401  pira  hown
        0      0.0      0.0    4500.0   47   6765.0  ...        0     0     0     0     1
        1   6215.0   1015.0   22390.0   36  28452.0  ...        0     0     0     0     1
        2      0.0  -2000.0   -2000.0   37   3300.0  ...        0     0     0     0     0
        3  15000.0  15000.0  155000.0   58  52590.0  ...        1     0     0     0     1
        4      0.0      0.0   58000.0   32  21804.0  ...        0     0     0     0     1
        [5 rows x 14 columns]

    .. code-tab:: r R

        > # R-code here
        > a=5

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

DoubleMLData from pandas DataFrames
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~doubleml.double_ml_data.DoubleMLData` class serves as data-backend and can be initialized from a :py:class:`pandas.DataFrame` by
specifying the column ``y_col='net_tfa'`` serving as outcome variable :math:`Y`, the column(s) ``d_cols = 'e401'``
serving as treatment variable :math:`D` and the columns ``x_cols=['age', 'inc', 'educ', 'fsize', 'marr', 'twoearn', 'db', 'pira', 'hown']``
specifying the confounders.

.. tabs::

    .. code-tab:: py

        >>> from doubleml import DoubleMLData

        >>> # Specify the data and the variables for the causal model
        >>> obj_dml_data_401k = DoubleMLData(df_401k,
        >>>                                  y_col='net_tfa',
        >>>                                  d_cols='e401',
        >>>                                  x_cols=['age', 'inc', 'educ', 'fsize', 'marr',
        >>>                                          'twoearn', 'db', 'pira', 'hown'])
        >>> print(obj_dml_data_401k)
        === DoubleMLData Object ===
        y_col: net_tfa
        d_cols: ['e401']
        x_cols: ['age', 'inc', 'educ', 'fsize', 'marr', 'twoearn', 'db', 'pira', 'hown']
        z_col: None
        data:
         <class 'pandas.core.frame.DataFrame'>
        Int64Index: 9915 entries, 0 to 9914
        Columns: 14 entries, nifa to hown
        dtypes: float32(4), int8(10)
        memory usage: 329.2 KB

    .. code-tab:: r R

        > # R-code here
        > a=5


DoubleMLData from numpy arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To introduce the :py:class:`numpy.ndarray` interface we generate a data set consisting of confounding variables ``X``, an outcome
variable ``y`` and a treatment variable ``d``


.. tabs::

    .. code-tab:: py

        >>> import numpy as np

        >>> # Generate data
        >>> n_obs = 500
        >>> n_vars = 100
        >>> theta = 3
        >>> X = np.random.normal(size=(n_obs, n_vars))
        >>> d = np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))
        >>> y = theta * d + np.dot(X[:, :3], np.array([5, 5, 5])) + np.random.standard_normal(size=(n_obs,))

    .. code-tab:: r R

        > # R-code here
        > a=5

To specify the data and the variables for the causal model from :py:class:`numpy.ndarray` we call

.. tabs::

    .. code-tab:: py

        >>> from doubleml import DoubleMLData

        >>> obj_dml_data_sim = DoubleMLData.from_arrays(X, y, d)
        >>> print(obj_dml_data_sim)
        === DoubleMLData Object ===
        y_col: y
        d_cols: ['d']
        x_cols: ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23', 'X24', 'X25', 'X26', 'X27', 'X28', 'X29', 'X30', 'X31', 'X32', 'X33', 'X34', 'X35', 'X36', 'X37', 'X38', 'X39', 'X40', 'X41', 'X42', 'X43', 'X44', 'X45', 'X46', 'X47', 'X48', 'X49', 'X50', 'X51', 'X52', 'X53', 'X54', 'X55', 'X56', 'X57', 'X58', 'X59', 'X60', 'X61', 'X62', 'X63', 'X64', 'X65', 'X66', 'X67', 'X68', 'X69', 'X70', 'X71', 'X72', 'X73', 'X74', 'X75', 'X76', 'X77', 'X78', 'X79', 'X80', 'X81', 'X82', 'X83', 'X84', 'X85', 'X86', 'X87', 'X88', 'X89', 'X90', 'X91', 'X92', 'X93', 'X94', 'X95', 'X96', 'X97', 'X98', 'X99', 'X100']
        z_col: None
        data:
         <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 500 entries, 0 to 499
        Columns: 102 entries, X1 to d
        dtypes: float64(102)
        memory usage: 398.6 KB

    .. code-tab:: r R

        > # R-code here
        > a=5

Estimate a causal model with double/debiased machine learning
-------------------------------------------------------------

Machine learners to estimate the nuisance models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To estimate our partially linear regression (PLR) model with the double machine learning algorithm, we first have to
specify machine learners to estimate :math:`m_0` and :math:`g_0`. For the 401(k) data we use
:py:class:`~sklearn.ensemble.RandomForestRegressor` from :py:mod:`sklearn.ensemble`
and for our simulated data from a sparse linear model we use
:py:class:`~sklearn.linear_model.Lasso` from :py:mod:`sklearn.linear_model`.

.. tabs::

    .. code-tab:: py

        >>> from sklearn.base import clone
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from sklearn.linear_model import Lasso

        >>> learner = RandomForestRegressor(max_depth=2, n_estimators=100)
        >>> ml_learners_401k = {'ml_m': clone(learner),
        >>>                     'ml_g': clone(learner)}

        >>> learner = Lasso(alpha=np.sqrt(np.log(n_vars)/(n_obs)))
        >>> ml_learners_sim = {'ml_m': clone(learner),
        >>>                    'ml_g': clone(learner)}

    .. code-tab:: r R

        > # R-code here
        > a=5

Cross-fitting, DML algorithms and Neyman-orthogonal score functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When initializing the object for PLR models :class:`~doubleml.double_ml_plr.DoubleMLPLR`, we can further set parameters specifying the
resampling: The number of folds used for cross-fitting ``n_folds`` (defaults to ``n_folds = 5``) as well as the number
of repetitions when applying repeated cross-fitting ``n_rep_cross_fit`` (defaults to ``n_rep_cross_fit = 1``).
Additionally, one can choose between the algorithms ``'dml1'`` and  ``'dml2'`` via ``dml_procedure``. Depending on the
causal model, one can further choose between different Neyman-orthogonal score / moment functions.

DoubleMLPLR: Double/debiased machine learning for partially linear regression models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We now initialize :class:`~doubleml.double_ml_plr.DoubleMLPLR` objects for our examples using default parameters


.. tabs::

    .. code-tab:: py

        >>> from doubleml import DoubleMLPLR
        >>> obj_dml_plr_401k = DoubleMLPLR(obj_dml_data_401k, ml_learners_401k)
        >>> obj_dml_plr_sim = DoubleMLPLR(obj_dml_data_sim, ml_learners_sim)

    .. code-tab:: r R

        > # R-code here
        > a=5

Estimate double/debiased machine learning models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The models are estimated by calling the ``fit()`` method and we can inspect the estimated treatment effect using the
``summary`` property.


.. tabs::

    .. code-tab:: py

        >>> obj_dml_plr_401k.fit()
        >>> print(obj_dml_plr_401k.summary)
                     coef   std err         t         P>|t|        2.5 %        97.5 %
        e401  9842.765039  1367.591  7.197155  6.148165e-13  7162.335933  12523.194145

        >>> obj_dml_plr_sim.fit()
        >>> print(obj_dml_plr_sim.summary)
               coef   std err          t  P>|t|    2.5 %    97.5 %
        d  2.982792  0.061354  48.615984    0.0  2.86254  3.103044

    .. code-tab:: r R

        > # R-code here
        > a=5
