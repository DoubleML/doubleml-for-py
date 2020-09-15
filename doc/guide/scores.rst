Score functions
---------------

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

    \psi(W; \theta, \eta) = \psi_a(W; \eta) \theta + \psi_b(W; \eta).

Hence the estimator can be written as

.. math::

    \tilde{\theta}_0 = - \frac{\mathbb{E}_N[\psi_b(W; \eta)]}{\mathbb{E}_N[\psi_a(W; \eta)]}

Implementation of the score function and the estimate of the causal parameter
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

As an example we consider a partially linear regression model (PLR)
implemented in :class:`~doubleml.double_ml_plr.DoubleMLPLR`.

.. ipython:: python

    import doubleml as dml
    from doubleml.datasets import make_plr_data
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.base import clone

    learner = RandomForestRegressor(max_depth=2, n_estimators=10)
    ml_learners = {'ml_m': clone(learner), 'ml_g': clone(learner)}
    data = make_plr_data()
    obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
    dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_learners)
    dml_plr_obj.fit()

The :meth:`~doubleml.double_ml_plr.DoubleMLPLR.fit` method of :class:`~doubleml.double_ml_plr.DoubleMLPLR`
stores the estimate :math:`\tilde{\theta}_0` in its ``coef`` attribute.

.. ipython:: python

    print(dml_plr_obj.coef)

The values of the score function components :math:`\psi_a(W_i; \hat{\eta}_0)` and :math:`\psi_b(W_i; \hat{\eta}_0)`
are stored in the attributes ``psi_a`` and ``psi_b``.
In the attribute ``psi`` the values of the score function :math:`\psi(W_i; \tilde{\theta}_0, \hat{\eta}_0)` are stored.

.. ipython:: python

    print(dml_plr_obj.psi[:5])

Implemented Neyman orthogonal score functions
+++++++++++++++++++++++++++++++++++++++++++++

Partially linear regression model (PLR)
***************************************

For the PLR model implemented in :class:`~doubleml.double_ml_plr.DoubleMLPLR` one can choose between
``score='IV-type'`` and ``score='DML2018'``.

``score='IV-type'`` implements the score function:

.. math::

    \psi(W; \theta, \eta) &:= [Y - D \theta - g(X)] [D - m(X)]

    &= - D (D - m(X)) \theta + (Y - g(X)) (D - m(X))

    &= \psi_a(W; \eta) \theta + \psi_b(W; \eta)

with :math:`\eta=(g,m)` and where the components of the linear score are

.. math::

    \psi_a(W; \eta) &=  - D (D - m(X)),

    \psi_b(W; \eta) &= (Y - g(X)) (D - m(X)).

``score='DML2018'`` implements the score function:

.. math::

    \psi(W; \theta, \eta) &:= [Y - \ell(X) - \theta (D - m(X))] [D - m(X)]

    &= - (D - m(X)) (D - m(X)) \theta + (Y - \ell(X)) (D - m(X))

    &= \psi_a(W; \eta) \theta + \psi_b(W; \eta)

with :math:`\eta=(\ell,m)` and where the components of the linear score are

.. math::

    \psi_a(W; \eta) &=  - (D - m(X)) (D - m(X)),

    \psi_b(W; \eta) &= (Y - \ell(X)) (D - m(X)).


Partially linear IV regression model (PLIV)
*******************************************

For the PLIV model implemented in :class:`~doubleml.double_ml_pliv.DoubleMLPLIV`
we employ for ``score='DML2018'`` the score function:

.. math::

    \psi(W; \theta, \eta) &:= [Y - \ell(X) - \theta (D - r(X))] [Z - m(X)]

    &= - (D - r(X)) (Z - m(X)) \theta + (Y - \ell(X)) (Z - m(X))

    &= \psi_a(W; \eta) \theta + \psi_b(W; \eta)

with :math:`\eta=(\ell, m, r)` and where the components of the linear score are

.. math::

    \psi_a(W; \eta) &=  - (D - r(X)) (Z - m(X)),

    \psi_b(W; \eta) &= (Y - \ell(X)) (Z - m(X)).

Interactive regression model (IRM)
**********************************

For the IRM model implemented in :class:`~doubleml.double_ml_irm.DoubleMLIRM` one can choose between
``score='ATE'`` and ``score='ATTE'``.

``score='ATE'`` implements the score function:

.. math::

    \psi(W; \theta, \eta) &:= g(1,X) - g(0,X) + \frac{D (Y - g(1,X))}{m(X)} - \frac{(1 - D)(Y - g(0,X))}{1 - m(x)} - \theta

    &= \psi_a(W; \eta) \theta + \psi_b(W; \eta)

with :math:`\eta=(g,m)` and where the components of the linear score are

.. math::

    \psi_a(W; \eta) &=  - 1,

    \psi_b(W; \eta) &= g(1,X) - g(0,X) + \frac{D (Y - g(1,X))}{m(X)} - \frac{(1 - D)(Y - g(0,X))}{1 - m(x)}.

``score='ATTE'`` implements the score function:

.. math::

    \psi(W; \theta, \eta) &:= \frac{D (Y - g(0,X))}{p} - \frac{m(X) (1 - D) (Y - g(0,X))}{p(1 - m(x))} - \frac{D}{p} \theta

    &= \psi_a(W; \eta) \theta + \psi_b(W; \eta)

with :math:`\eta=(g, m, p)` and where the components of the linear score are

.. math::

    \psi_a(W; \eta) &=  - \frac{D}{p},

    \psi_b(W; \eta) &= \frac{D (Y - g(0,X))}{p} - \frac{m(X) (1 - D) (Y - g(0,X))}{p(1 - m(X))}.


Interactive IV model (IIVM)
***************************

For the IIVM model implemented in :class:`~doubleml.double_ml_iivm.DoubleMLIIVM`
we employ for ``score='LATE'`` the score function:

``score='LATE'`` implements the score function:

.. math::

    \psi(W; \theta, \eta) :=\; &\mu(1,X) - \mu(0,X)
    + \frac{Z (Y - \mu(1,X))}{p(X)} - \frac{(1 - Z)(Y - \mu(0,X))}{1 - p(x)}

    &- \bigg(m(1,X) - m(0,X) + \frac{Z (D - m(1,X))}{p(X)} - \frac{(1 - Z)(D - m(0,X))}{1 - p(x)} \bigg) \theta

    =\; &\psi_a(W; \eta) \theta + \psi_b(W; \eta)

with :math:`\eta=(g,m)` and where the components of the linear score are

.. math::

    \psi_a(W; \eta) &=  - \bigg(m(1,X) - m(0,X) + \frac{Z (D - m(1,X))}{p(X)} - \frac{(1 - Z)(D - m(0,X))}{1 - p(x)} \bigg),

    \psi_b(W; \eta) &= \mu(1,X) - \mu(0,X) + \frac{Z (Y - \mu(1,X))}{p(X)} - \frac{(1 - Z)(Y - \mu(0,X))}{1 - p(x)}.

Specifying alternative score functions via callables
++++++++++++++++++++++++++++++++++++++++++++++++++++

Via callables user-written score functions can be used.
For the PLR model implemented in :class:`~doubleml.double_ml_plr.DoubleMLPLR` an alternative score function can be
set via ``score``.
Choose a callable object / function with signature ``score(y, d, g_hat, m_hat, smpls)`` which returns
the two score components :math:`\psi_a()` and :math:`\psi_b()`.

For example, the non-orthogonal score function

.. math::

    \psi(W; \theta, \eta) = [Y - D \theta - g(X)] D

can be obtained with

.. ipython:: python

    def non_orth_score(y, d, g_hat, m_hat, smpls):
        u_hat = y - g_hat
        psi_a = -np.multiply(d, d)
        psi_b = np.multiply(d, u_hat)
        return psi_a, psi_b

Use :class:`~doubleml.double_ml_plr.DoubleMLPLR` with ``inf_model=non_orth_score`` in order to obtain the estimator

.. math::

    \tilde{\theta}_0 = - \frac{\mathbb{E}_N[D (Y-g(X))]}{\mathbb{E}_N[D^2]}

when applying :meth:`~doubleml.double_ml_plr.DoubleMLPLR.fit`.
Note that this estimate will in general be prone to a regularization bias, see also :ref:`bias_non_orth`.

