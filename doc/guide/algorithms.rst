.. _algorithms:

Double machine learning algorithms
----------------------------------

The DoubleML package comes with two different algorithms to obtain DML estimates.

Algorithm DML1
++++++++++++++

The algorithm ``dml_procedure='dml1'`` can be summarized as

1. **Inputs:** Choose a model (PLR, PLIV, IRM, IIVM), provide data :math:`(W_i)_{i=1}^{N}`, a Neyman-orthogonal score function :math:`\psi(W; \theta, \eta)` and specify machine learning method(s) for the nuisance function(s) :math:`\eta`.

2. **Train ML predictors on folds:** Take a :math:`K`-fold random partition :math:`(I_k)_{k=1}^{K}` of observation indices :math:`[N] = \lbrace 1, \ldots, N\rbrace` such that the size of each fold :math:`I_k` is :math:`n=N/K`. For each :math:`k \in [K] = \lbrace 1, \ldots, K]`, construct a high-quality machine learning estimator

    .. math::

        \hat{\eta}_{0,k} = \hat{\eta}_{0,k}\big((W_i)_{i\not\in I_k}\big)

    of :math:`\eta_0`, where :math:`x \mapsto \hat{\eta}_{0,k}(x)` depends only on the subset of data :math:`(W_i)_{i\not\in I_k}`.

3. **Estimate causal parameter:** For each :math:`k \in [K]`, construct the estimator :math:`\check{\theta}_{0,k}` as the solution to the equation

    .. math::

        \frac{1}{n} \sum_{i \in I_k} \psi(W_i; \check{\theta}_{0,k}, \hat{\eta}_{0,k}) = 0.

    The estimate of the causal parameter is obtain via aggregation

    .. math::

        \tilde{\theta}_0 = \sum_{k=1}^{K} \check{\theta}_{0,k}.


4. **Outputs:** The estimate of the causal parameter :math:`\tilde{\theta}_0` as well as the values of the evaluated score function are returned.

Algorithm DML2
++++++++++++++

The algorithm ``dml_procedure='dml2'`` can be summarized as

1. **Inputs:** Choose a model (PLR, PLIV, IRM, IIVM), provide data :math:`(W_i)_{i=1}^{N}`, a Neyman-orthogonal score function :math:`\psi(W; \theta, \eta)` and specify machine learning method(s) for the nuisance function(s) :math:`\eta`.

2. **Train ML predictors on folds:** Take a :math:`K`-fold random partition :math:`(I_k)_{k=1}^{K}` of observation indices :math:`[N] = \lbrace 1, \ldots, N\rbrace` such that the size of each fold :math:`I_k` is :math:`n=N/K`. For each :math:`k \in [K] = \lbrace 1, \ldots, K]`, construct a high-quality machine learning estimator

    .. math::

        \hat{\eta}_{0,k} = \hat{\eta}_{0,k}\big((W_i)_{i\not\in I_k}\big)

    of :math:`\eta_0`, where :math:`x \mapsto \hat{\eta}_{0,k}(x)` depends only on the subset of data :math:`(W_i)_{i\not\in I_k}`.

3. **Estimate causal parameter:** Construct the estimator for the causal parameter :math:`\tilde{\theta}_0` as the solution to the equation

    .. math::

        \frac{1}{N} \sum_{k=1}^{K} \sum_{i \in I_k} \psi(W_i; \tilde{\theta}_0, \hat{\eta}_{0,k}) = 0.


4. **Outputs:** The estimate of the causal parameter :math:`\tilde{\theta}_0` as well as the values of the evaluate score function are returned.

Implementation of the double machine learning algorithms
++++++++++++++++++++++++++++++++++++++++++++++++++++++++

As an example we consider a partially linear regression model (PLR)
implemented in ``DoubleMLPLR``.
The DML algorithm can be selected via parameter ``dml_procedure='dml1'`` vs. ``dml_procedure='dml2'``.

.. tabbed:: Python

    .. ipython:: python

        import doubleml as dml
        from doubleml.datasets import make_plr_CCDDHNR2018
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.base import clone

        np.random.seed(3141)
        learner = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
        ml_g = clone(learner)
        ml_m = clone(learner)
        data = make_plr_CCDDHNR2018(alpha=0.5, return_type='DataFrame')
        obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m, dml_procedure='dml1')
        dml_plr_obj.fit();

.. tabbed:: R

    .. jupyter-execute::

        library(DoubleML)
        library(mlr3)
        library(mlr3learners)
        library(data.table)
        lgr::get_logger("mlr3")$set_threshold("warn")

        learner = lrn("regr.ranger", num.trees = 100, mtry = 20, min.node.size = 2, max.depth = 5)
        ml_g = learner$clone()
        ml_m = learner$clone()
        set.seed(3141)
        data = make_plr_CCDDHNR2018(alpha=0.5, return_type='data.table')
        obj_dml_data = DoubleMLData$new(data, y_col="y", d_cols="d")
        dml_plr_obj = DoubleMLPLR$new(obj_dml_data, ml_g, ml_m, dml_procedure="dml1")
        dml_plr_obj$fit()


The ``fit()`` method of ``DoubleMLPLR``
stores the estimate :math:`\tilde{\theta}_0` in its ``coef`` attribute.

.. tabbed:: Python

    .. ipython:: python

        dml_plr_obj.coef

.. tabbed:: R

    .. jupyter-execute::

        dml_plr_obj$coef

Let :math:`k(i) = \lbrace k: i \in I_k \rbrace`.
The values of the score function :math:`(\psi(W_i; \tilde{\theta}_0, \hat{\eta}_{0,k(i)}))_{i \in [N]}`
are stored in the attribute ``psi``.


.. tabbed:: Python

    .. ipython:: python

        dml_plr_obj.psi[:5]

.. tabbed:: R

    .. jupyter-execute::

        dml_plr_obj$psi[1:5, ,1]


For the DML1 algorithm, the estimates for the different folds
:math:`\check{\theta}_{0,k}``, :math:`k \in [K]` are stored in attribute ``all_dml1_coef``.

.. tabbed:: Python

    .. ipython:: python

        dml_plr_obj.all_dml1_coef

.. tabbed:: R

    .. jupyter-execute::

        dml_plr_obj$all_dml1_coef

