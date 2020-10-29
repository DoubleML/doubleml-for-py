**Interactive IV regression (IIVM)** models take the form

.. math::

    Y = g_0(D, X) + \zeta, & &\mathbb{E}(\zeta | Z, X) = 0,

    Z = m_0(X) + V, & &\mathbb{E}(V | X) = 0,

where the treatment variable is binary, :math:`D \in \lbrace 0,1 \rbrace`
and the instrument is binary, :math:`Z \in \lbrace 0,1 \rbrace`.
Consider the functions :math:`g_0`, :math:`r_0` and :math:`m_0`, where :math:`g_0` maps the support of :math:`(Z,X)` to
:math:`\mathbb{R}` and :math:`r_0` and :math:`m_0` respectively map the support of :math:`(Z,X)` and :math:`X` to
:math:`(\varepsilon, 1-\varepsilon)` for some :math:`\varepsilon \in (0, 1/2)`, such that

.. math::

    Y = g_0(D, X) + \zeta, & &\mathbb{E}(\zeta | Z, X) = 0,

    D = r_0(D, X) + U, & &\mathbb{E}(U | Z, X) = 0,

    Z = m_0(X) + V, & &\mathbb{E}(V | X) = 0.

The target parameter of interest in this model is the local average treatment effect (LATE),

.. math::

    \theta_0 = \frac{\mathbb{E}[g(1, X)] - \mathbb{E}[g(0,X)]}{\mathbb{E}[r(1, X)] - \mathbb{E}[r(0,X)]}.