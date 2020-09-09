**Partially linear regression (PLR)** models take the form

.. math::

    Y = D \theta_0 + g_0(X) + \zeta, & &\mathbb{E}(\zeta | D,X) = 0,

    D = m_0(X) + V, & &\mathbb{E}(V | X) = 0,

where :math:`Y` is the outcome variable and :math:`D` is the policy variable of interest.
The high-dimensional vector :math:`X = (X_1, \ldots, X_p)` consists of other confounding covariates,
and :math:`\zeta` and :math:`V` are stochastic errors.