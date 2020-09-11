.. _dml-algo:

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
implemented in :class:`~doubleml.double_ml_plr.DoubleMLPLR`.
The DML algorithm can be selected via parameter ``dml_procedure='dml1'`` vs. ``dml_procedure='dml2'``.

.. tabs::

    .. code-tab:: py

        import doubleml as dml
        from doubleml.datasets import make_plr_data
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.base import clone

        learner = RandomForestRegressor(max_depth=2, n_estimators=10)
        ml_learners = {'ml_m': clone(learner), 'ml_g': clone(learner)}
        data = make_plr_data()
        obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_learners,
                                      dml_procedure='dml1')
        dml_plr_obj.fit()

    .. code-tab:: r R

        R

The :meth:`~doubleml.double_ml_plr.DoubleMLPLR.fit` method of :class:`~doubleml.double_ml_plr.DoubleMLPLR`
stores the estimate :math:`\tilde{\theta}_0` in its ``coef`` attribute.

.. tabs::

    .. code-tab:: py

        print(dml_plr_obj.coef)

    .. code-tab:: r R

        R

Let :math:`k(i) = \lbrace k: i \in I_k \rbrace`.
The values of the score function :math:`(\psi(W_i; \tilde{\theta}_0, \hat{\eta}_{0,k(i)}))_{i \in [N]}`
are stored in the attribute ``score``.


.. tabs::

    .. code-tab:: py

        print(dml_plr_obj.score[:5])

    .. code-tab:: r R

        R

For the DML1 algorithm, the estimates for the different folds
:math:`\check{\theta}_{0,k}``, :math:`k \in [K]` are stored in attribute ``_all_dml1_coef``.

.. tabs::

    .. code-tab:: py

        print(dml_plr_obj._all_dml1_coef)

    .. code-tab:: r R

        R

