**Interactive IV (IIVM)** models take the form

.. math::

    Y = g_0(D, X) + \zeta, & &\mathbb{E}(\zeta | Z, X) = 0,

    Z = m_0(X) + V, & &\mathbb{E}(V | X) = 0,

where the treatment variable is binary, :math:`D \in \lbrace 0,1 \rbrace`
and the instrument is binary, :math:`Z \in \lbrace 0,1 \rbrace`.
The target parameter of interest in this model is the local average treatment effect (LATE),

.. math::

    \theta_0 = \frac{\mathbb{E}[\mu(1, X)] - \mathbb{E}[\mu(0,X)]}{\mathbb{E}[m(1, X)] - \mathbb{E}[m(0,X)]}