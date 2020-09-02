**Interactive regression (IRM)** models take the form

.. math::

    Y = g_0(D, X) + U, & &\mathbb{E}(U | X, D) = 0,

    D = m_0(X) + V, & &\mathbb{E}(V | X) = 0,

where the treatment variable is binary, :math:`D \in \lbrace 0,1 \rbrace`.
We consider estimation of the average treatment effects when treatment effects are fully heterogeneous.
Target parameters of interest in this model are the average treatment effect (ATE),

.. math::

    \theta_0 = \mathbb{E}[g_0(1, X) - g_0(0,X)]

and the average treatment effect of the treated (ATTE),

.. math::

    \theta_0 = \mathbb{E}[g_0(1, X) - g_0(0,X) | D=1].
