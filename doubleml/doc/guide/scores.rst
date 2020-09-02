Score functions
---------------

Neyman orthogonal score functions
+++++++++++++++++++++++++++++++++

We use method-of-moments estimators for the target parameter :math:`\theta_0` based upon the empirical analog of the
moment condition

.. math::

    \mathbb{E}[ \psi(W; \theta_0, \eta_0)] = 0,

where we call :math:`\psi` the **score function**, :math:`W=(Y,D,X,Z)`,
:math:`\theta_0` is the parameter of interest and
:math:`\eta` denotes nuisance functions with population value :math:`\eta_0`.
We use score functions :math:`\psi(W; \theta, \eta)` that satisfy
:math:`\mathbb{E}[ \psi(W; \theta_0, \eta_0)] = 0` with :math:`\theta_0` being the unique solution
and that obey the **Neyman orthogonality condition**

.. math::

    \partial_{\eta} \mathbb{E}[ \psi(W; \theta_0, \eta)] \bigg|_{\eta=\eta_0} = 0.

The object-oriented (OOP) implementation of
:class:`~doubleml.double_ml_plr.DoubleMLPLR`,
:class:`~doubleml.double_ml_pliv.DoubleMLPLIV`,
:class:`~doubleml.double_ml_irm.DoubleMLIRM`,
and :class:`~doubleml.double_ml_iivm.DoubleMLIIVM`
uses the linearity of the score function in the parameter :math:`theta`

.. math::

    \psi(W; \theta, \eta) = \psi^a(W; \eta) \theta + \psi^b(W; \eta).

Hence the estimator can be written as

.. math::

    \tilde{\theta}_0 = - \frac{\mathbb{E}_N[\psi^b(W; \eta)]}{\mathbb{E}_N[\psi^a(W; \eta)]}

