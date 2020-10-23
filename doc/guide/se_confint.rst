.. _se_confint:

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
implemented in ``DoubleMLPLR``.

.. tabbed:: Python

    .. ipython:: python

        import doubleml as dml
        from doubleml.datasets import make_plr_CCDDHNR2018
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.base import clone

        learner = RandomForestRegressor(max_depth=2, n_estimators=10)
        ml_g = clone(learner)
        ml_m = clone(learner)
        data = make_plr_CCDDHNR2018(return_type='DataFrame')
        obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m)
        dml_plr_obj.fit();

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)

The ``fit()`` method of ``DoubleMLPLR``
stores the estimate :math:`\tilde{\theta}_0` in its ``coef`` attribute.

.. tabbed:: Python

    .. ipython:: python

        print(dml_plr_obj.coef)

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)

The asymptotic standard error :math:`\hat{\sigma}/\sqrt{N}` is stored in its ``se`` attribute.

.. tabbed:: Python

    .. ipython:: python

        print(dml_plr_obj.se)

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)

Additionally, the value of the :math:`t`-statistic and the corresponding p-value are provided in the attributes
``t_stat`` and ``pval``.

.. tabbed:: Python

    .. ipython:: python

        print(dml_plr_obj.t_stat)
        print(dml_plr_obj.pval)

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)

An overview of all these estimates, together with a 95 % confidence interval is stored in the attribute ``summary``.

.. tabbed:: Python

    .. ipython:: python

        print(dml_plr_obj.summary)

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)

A more detailed overview of the fitted model, its specifications and the summary can be obtained via the
string-representation of the object.

.. tabbed:: Python

    .. ipython:: python

        print(dml_plr_obj)

.. tabbed:: R

    .. jupyter-execute::

        X = c(1,4,5,6);
        Y = c(5,3,5,7);
        lm(Y~X)

.. TODO: Add a documentation of the ``se_reestimate`` option here (especially for DML1 algorithm).

Boostrap standard errors and joint confidence intervals
+++++++++++++++++++++++++++++++++++++++++++++++++++++++

The ``bootstrap()`` method provides an implementation of a multiplier bootstrap for double machine learning models.
For :math:`b=1, \ldots, B` weights :math:`\xi_{i, b}` are generated according to a normal (Gaussian) bootstrap, wild
bootstrap or exponential bootstrap.
The number of bootstrap samples is provided as input ``n_boot_rep`` and for ``method`` one can choose ``'Bayes'``,
``'normal'`` or ``'wild'``.
Based on the estimates of the standard errors given by

.. math::

    \sigma^2 := J_0^{-2} \mathbb{E}(\psi^2(W; \theta_0, \eta_0)),

    J_0 = \mathbb{E}(\psi_a(W; \eta_0)).

we obtain bootstrap coefficients :math:`\theta^*_b` and bootstrap t-statistics :math:`t^*_b`

.. math::

    \theta^*_b &= \frac{1}{\sqrt{N} \hat{J}_0}\sum_{k=1}^{K} \sum_{i \in I_k} \xi_{i, b} \cdot \psi(W_i; \tilde{\theta}_0, \hat{\eta}_{0,k}),

    t^*_b &= \frac{1}{\sqrt{N} \hat{J}_0 \hat{\sigma}} \sum_{k=1}^{K} \sum_{i \in I_k} \xi_{i, b} \cdot \psi(W_i; \tilde{\theta}_0, \hat{\eta}_{0,k}).


To demonstrate the bootstrap, we simulate data from a sparse partially linear regression model.
Then we estimate the PLR model and perform the multiplier bootstrap.
Joint confidence intervals based on the multiplier bootstrap are then obtained with the method ``confint()``
or do a multiple hypotheses testing adjustment of p-values from a high-dimensional model can be performed with the
method ``p_adjust``.

.. tabbed:: Python

    .. ipython:: python

        import doubleml as dml
        import numpy as np
        from sklearn.base import clone
        from sklearn.linear_model import Lasso

        # Simulate data
        np.random.seed(1234)
        n_obs = 500
        n_vars = 100
        X = np.random.normal(size=(n_obs, n_vars))
        theta = np.array([3., 3., 3.])
        y = np.dot(X[:, :3], theta) + np.random.standard_normal(size=(n_obs,))

        dml_data = dml.DoubleMLData.from_arrays(X[:, 10:], y, X[:, :10])

        learner = Lasso(alpha=np.sqrt(np.log(n_vars)/(n_obs)))
        ml_g = clone(learner)
        ml_m = clone(learner)
        dml_plr = dml.DoubleMLPLR(dml_data, ml_g, ml_m)

        print(dml_plr.fit().bootstrap().confint(joint=True))
        print(dml_plr.p_adjust())

