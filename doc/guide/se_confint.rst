.. _se-confint:

Variance estimation, confidence intervals and boostrap standard errors
----------------------------------------------------------------------

Variance estimation
+++++++++++++++++++

Under regularity conditions the estimator :math:`\tilde{\theta}_0` concentrates in a :math:`1/\sqrt(N)`-neighborhood
of :math:`\theta_0` and the sampling error :math:`\sqrt(N)(\tilde{\theta}_0 - \theta_0)` is approximately normal

.. math::

    \sqrt(N)(\tilde{\theta}_0 - \theta_0) \leadsto N(o, \sigma^2),

with mean zero and variance given by

.. math::

    \sigma^2 := J_0^{-2} \mathbb{E}(\psi^2(W; \theta_0, \eta_0)),

    J_0 = \mathbb{E}(\psi_a(W; \eta_0)).

Estimates of the variance are obtained by

.. math::

    \hat{\sigma}^2 &= \hat{J}_0^{-2} \frac{1}{N} \sum_{k=1}^{K} \sum_{i \in I_k} \big[\psi(W_i; \tilde{\theta}_0, \hat{\eta}_{0,k})\big]^2,

    \hat{J}_0 &= \frac{1}{N} \sum_{k=1}^{K} \sum_{i \in I_k} \psi_a(W_i; \hat{\eta}_{0,k}).

An approximate confidence interval is given by

.. math::

    \big[\tilde{\theta}_0 \pm \Phi^{-1}(1 - \alpha/2) \hat{\sigma} / \sqrt{N}].

As an example we consider a partially linear regression model (PLR)
implemented in :class:`~doubleml.double_ml_plr.DoubleMLPLR`.

.. tabbed:: Python

    .. ipython:: python

        import doubleml as dml
        from doubleml.datasets import make_plr_data
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.base import clone

        learner = RandomForestRegressor(max_depth=2, n_estimators=10)
        ml_g = clone(learner)
        ml_m = clone(learner)
        data = make_plr_data()
        obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m)
        dml_plr_obj.fit()

.. tabbed:: R

    .. code-block:: R

        > # R-code here
        > a=5

The :meth:`~doubleml.double_ml_plr.DoubleMLPLR.fit` method of :class:`~doubleml.double_ml_plr.DoubleMLPLR`
stores the estimate :math:`\tilde{\theta}_0` in its ``coef`` attribute.

.. tabbed:: Python

    .. ipython:: python

        print(dml_plr_obj.coef)

.. tabbed:: R

    .. code-block:: R

        > # R-code here
        > a=5

The asymptotic standard error :math:`\hat{\sigma}/\sqrt{N}` is stored in its ``se`` attribute.

.. tabbed:: Python

    .. ipython:: python

        print(dml_plr_obj.se)

.. tabbed:: R

    .. code-block:: R

        > # R-code here
        > a=5

Additionally, the value of the :math:`t`-statistic and the corresponding p-value are provided in the attributes
``t_stat`` and ``pval``.

.. tabbed:: Python

    .. ipython:: python

        print(dml_plr_obj.t_stat)
        print(dml_plr_obj.pval)

.. tabbed:: R

    .. code-block:: R

        > # R-code here
        > a=5

An overview of all these estimates, together with a 95 % confidence interval is stored in the attribute ``summary``.

.. tabbed:: Python

    .. ipython:: python

        print(dml_plr_obj.summary)

.. tabbed:: R

    .. code-block:: R

        > # R-code here
        > a=5

.. TODO: Add a documentation of the ``se_reestimate`` option here (especially for DML1 algorithm).

Boostrap standard errors and joint confidence intervals
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. TODO Document the multiplier bootstrap and joint confidence intervals.
