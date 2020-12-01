.. _scores:

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

An integral component for the object-oriented (OOP) implementation of
``DoubleMLPLR``,
``DoubleMLPLIV``,
``DoubleMLIRM``,
and ``DoubleMLIIVM``
is the linearity of the score function in the parameter :math:`\theta`

.. math::

    \psi(W; \theta, \eta) = \psi_a(W; \eta) \theta + \psi_b(W; \eta).

Hence the estimator can be written as

.. math::

    \tilde{\theta}_0 = - \frac{\mathbb{E}_N[\psi_b(W; \eta)]}{\mathbb{E}_N[\psi_a(W; \eta)]}.

The linearity of the score function in the parameter :math:`\theta` allows the implementation of key components in a very
general way.
The methods and algorithms to estimate the causal parameters, to estimate their standard errors, to perform a multiplier
bootstrap, to obtain confidence intervals and many more are implemented in the abstract base class ``DoubleML``.
The object-oriented architecture therefore allows for easy extension to new model classes for double machine learning.
This is doable with very minor effort whenever the linearity of the score function is satisfied.

Implementation of the score function and the estimate of the causal parameter
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

As an example we consider a partially linear regression model (PLR)
implemented in ``DoubleMLPLR``.

.. tabbed:: Python

    .. ipython:: python

        import doubleml as dml
        from doubleml.datasets import make_plr_CCDDHNR2018
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.base import clone

        np.random.seed(3141)
        learner = RandomForestRegressor(n_estimators=100, max_features=20, max_depth=5, min_samples_leaf=2)
        ml_g = clone(learner)
        ml_m = clone(learner)
        data = make_plr_CCDDHNR2018(alpha=0.5, return_type='DataFrame')
        obj_dml_data = dml.DoubleMLData(data, 'y', 'd')
        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m)
        dml_plr_obj.fit();
        print(dml_plr_obj)

.. tabbed:: R

    .. jupyter-execute::
        :raises:

        library(DoubleML)
        library(mlr3)
        library(mlr3learners)
        library(data.table)
        lgr::get_logger("mlr3")$set_threshold("warn")

        learner = lrn("regr.ranger", num.trees = 100, mtry = 20, min.node.size = 2, max.depth = 5)
        ml_g = learner$clone()
        ml_m = learner$clone()
        set.seed(3141)
        data = make_plr_CCDDHNR2018(alpha=0.5, return_type='data.table')
        obj_dml_data = DoubleMLData$new(data, y_col="y", d_cols="d")
        dml_plr_obj = DoubleMLPLR$new(obj_dml_data, ml_g, ml_m)
        dml_plr_obj$fit()
        print(dml_plr_obj)

The ``fit()`` method of ``DoubleMLPLR``
stores the estimate :math:`\tilde{\theta}_0` in its ``coef`` attribute.

.. tabbed:: Python

    .. ipython:: python

        print(dml_plr_obj.coef)

.. tabbed:: R

    .. jupyter-execute::
        :raises:

        print(dml_plr_obj$coef)

The values of the score function components :math:`\psi_a(W_i; \hat{\eta}_0)` and :math:`\psi_b(W_i; \hat{\eta}_0)`
are stored in the attributes ``psi_a`` and ``psi_b``.
In the attribute ``psi`` the values of the score function :math:`\psi(W_i; \tilde{\theta}_0, \hat{\eta}_0)` are stored.

.. tabbed:: Python

    .. ipython:: python

        print(dml_plr_obj.psi[:5])

.. tabbed:: R

    .. jupyter-execute::
        :raises:

        print(dml_plr_obj$psi[1:5, ,1])


Implemented Neyman orthogonal score functions
+++++++++++++++++++++++++++++++++++++++++++++

Partially linear regression model (PLR)
***************************************

For the PLR model implemented in ``DoubleMLPLR`` one can choose between
``score='IV-type'`` and ``score='partialling out'``.

``score='IV-type'`` implements the score function:

.. math::

    \psi(W; \theta, \eta) &:= [Y - D \theta - g(X)] [D - m(X)]

    &= - D (D - m(X)) \theta + (Y - g(X)) (D - m(X))

    &= \psi_a(W; \eta) \theta + \psi_b(W; \eta)

with :math:`\eta=(g,m)` and where the components of the linear score are

.. math::

    \psi_a(W; \eta) &=  - D (D - m(X)),

    \psi_b(W; \eta) &= (Y - g(X)) (D - m(X)).

``score='partialling out'`` implements the score function:

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

For the PLIV model implemented in ``DoubleMLPLIV``
we employ for ``score='partialling out'`` the score function:

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

For the IRM model implemented in ``DoubleMLIRM`` one can choose between
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

For the IIVM model implemented in ``DoubleMLIIVM``
we employ for ``score='LATE'`` the score function:

``score='LATE'`` implements the score function:

.. math::

    \psi(W; \theta, \eta) :=\; &g(1,X) - g(0,X)
    + \frac{Z (Y - g(1,X))}{m(X)} - \frac{(1 - Z)(Y - g(0,X))}{1 - m(x)}

    &- \bigg(r(1,X) - r(0,X) + \frac{Z (D - r(1,X))}{m(X)} - \frac{(1 - Z)(D - r(0,X))}{1 - m(x)} \bigg) \theta

    =\; &\psi_a(W; \eta) \theta + \psi_b(W; \eta)

with :math:`\eta=(g, m, r)` and where the components of the linear score are

.. math::

    \psi_a(W; \eta) &=  - \bigg(r(1,X) - r(0,X) + \frac{Z (D - r(1,X))}{m(X)} - \frac{(1 - Z)(D - r(0,X))}{1 - m(x)} \bigg),

    \psi_b(W; \eta) &= g(1,X) - g(0,X) + \frac{Z (Y - g(1,X))}{m(X)} - \frac{(1 - Z)(Y - g(0,X))}{1 - m(x)}.

Specifying alternative score functions via callables
++++++++++++++++++++++++++++++++++++++++++++++++++++

Via callables user-written score functions can be used.
This functionality is at the moment only implemented for specific model classes in Python.
For the PLR model implemented in ``DoubleMLPLR`` an alternative score function can be
set via ``score``.
Choose a callable object / function with signature ``score(y, d, g_hat, m_hat, smpls)`` which returns
the two score components :math:`\psi_a()` and :math:`\psi_b()`.

For example, the non-orthogonal score function

.. math::

    \psi(W; \theta, \eta) = [Y - D \theta - g(X)] D

can be obtained with

.. tabbed:: Python

    .. ipython:: python

        import numpy as np

        def non_orth_score(y, d, g_hat, m_hat, smpls):
            u_hat = y - g_hat
            psi_a = -np.multiply(d, d)
            psi_b = np.multiply(d, u_hat)
            return psi_a, psi_b

.. tabbed:: R

    .. jupyter-execute::
        :raises:

        non_orth_score = function(y, d, g_hat, m_hat, smpls) {
            u_hat = y - g_hat
            psi_a = -1*d*d
            psi_b = d*u_hat
            psis = list(psi_a = psi_a, psi_b = psi_b)
            return(psis)
        }

Use ``DoubleMLPLR`` with ``inf_model=non_orth_score`` in order to obtain the estimator

.. math::

    \tilde{\theta}_0 = - \frac{\mathbb{E}_N[D (Y-g(X))]}{\mathbb{E}_N[D^2]}

when applying ``fit()``.
Note that this estimate will in general be prone to a regularization bias, see also :ref:`bias_non_orth`.

