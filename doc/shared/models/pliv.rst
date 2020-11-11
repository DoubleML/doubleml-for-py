**Partially linear IV regression (PLIV)** models take the form

.. math::

    Y - D \theta_0 =  g_0(X) + \zeta, & &\mathbb{E}(\zeta | Z, X) = 0,

    Z = m_0(X) + V, & &\mathbb{E}(V | X) = 0.

where :math:`Y` is the outcome variable, :math:`D` is the policy variable of interest and :math:`Z`
denotes one or multiple instrumental variables. The high-dimensional vector
:math:`X = (X_1, \ldots, X_p)` consists of other confounding covariates, and :math:`\zeta` and
:math:`V` are stochastic errors.