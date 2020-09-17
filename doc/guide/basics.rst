The basics of double/debiased machine learning
----------------------------------------------

Data generating process
+++++++++++++++++++++++

We consider the following partially linear model

.. math::

    Y_i = D_i \theta + g(X_i) + \zeta,

    D_i = m(X_i) + V,

with :math:`\zeta, V \sim \mathcal{N}(0,1)` and :math:`X \sim \mathcal{N}_{p}(0, \Sigma)`.
The variance-covariance matrix :math:`\Sigma` of the :math:`p`-dimensional confounders :math:`X` is a Toeplitz-matrix
with diagonal elements :math:`0.7^i`.
The true parameter :math:`\theta` is set to :math:`0.5`.
The non-linear functions :math:`g()` and :math:`m()` are chosen as

.. math::

    g(X_i) = \frac{\exp(X_{i1})}{1+\exp(X_{i1})} + \frac{1}{4} X_{i3},

    m(X_i) = \frac{\exp(X_{i3})}{1+\exp(X_{i3})}

.. tabbed:: Python

    .. ipython:: python

        import numpy as np
        from scipy.linalg import toeplitz

        def generate_data(n_obs, n_vars):
            cov_mat = toeplitz([np.power(0.7, k) for k in range(n_vars)])
            a_1 = 0.25
            b_1 = 0.25
            alpha = 0.5
            x = np.random.multivariate_normal(np.zeros(n_vars), cov_mat, size=[n_obs,])
            d = x[:,0] + a_1 * np.divide(np.exp(x[:,2]), 1 + np.exp(x[:,2])) \
                + np.random.standard_normal(size=[n_obs,])
            y = alpha * d + np.divide(np.exp(x[:,2]), 1 + np.exp(x[:,2])) \
                + b_1 * x[:,2] + np.random.standard_normal(size=[n_obs,])
            return x, y, d

        np.random.seed(1234)
        n_rep = 1000
        n_obs = 500
        n_vars = 20

        data = list()

        for i_rep in range(n_rep):
            (x, y, d) = generate_data(n_obs, n_vars)
            data.append((x, y, d))

.. tabbed:: R

    .. code-block:: R

        > # R-code here
        > a=5


OLS Estimation
++++++++++++++

A naive OLS regression of :math:`Y` on :math:`D` produces a significant bias.

.. tabbed:: Python

    .. ipython:: python

        from sklearn.linear_model import LinearRegression
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()
        colors = sns.color_palette()

        def est_ols(y, X):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            ols = LinearRegression(fit_intercept=False)
            results = ols.fit(X, y)
            theta = results.coef_
            return theta

        theta_ols = np.zeros(n_rep)
        for i_rep in range(n_rep):
            (x, y, d) = data[i_rep]
            theta_ols[i_rep] = est_ols(y, d)

        ax = sns.kdeplot(theta_ols, shade=True, color=colors[0])
        @savefig ols.png width=5in
        ax.axvline(0.5, color='k', label='True theta');

.. tabbed:: R

    .. code-block:: R

        > # R-code here
        > a=5


Regularization Bias in Simple ML-Approaches
+++++++++++++++++++++++++++++++++++++++++++

A simple ML approach is given by randomly splitting the sample into two parts.
On the auxiliary sample :math:`g(X)` is estimated with an ML method.
Given the estimate :math:`\hat{g}(X)`, the final estimate of :math:`\theta` is obtained as (:math:`n=N/2`)


.. math::

    \hat{\theta} = \left(\frac{1}{n} \sum_{i\in I} D_i^2\right)^{-1} \frac{1}{n} \sum_{i\in I} D_i (Y_i - \hat{g}(X_i))

.. tabbed:: Python

    .. ipython:: python

        def non_orth_score(y, d, g_hat, m_hat, smpls):
            u_hat = y - g_hat
            psi_a = -np.multiply(d, d)
            psi_b = np.multiply(d, u_hat)
            return psi_a, psi_b

    .. ipython:: python

        from doubleml import DoubleMLData
        from doubleml import DoubleMLPLR
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.base import clone

        learner = RandomForestRegressor(max_depth=2, n_estimators=10)
        ml_m = clone(learner)
        ml_g = clone(learner)
        theta_nonorth = np.zeros(n_rep)
        for i_rep in range(n_rep):
            (x, y, d) = data[i_rep]
            obj_dml_data = DoubleMLData.from_arrays(x, y, d)
            obj_dml_plr_nonorth = DoubleMLPLR(obj_dml_data,
                                              ml_m, ml_g,
                                              n_folds=2,
                                              apply_cross_fitting=False,
                                              score=non_orth_score)
            obj_dml_plr_nonorth.fit()
            theta_nonorth[i_rep] = obj_dml_plr_nonorth.coef[0]

        ax = sns.kdeplot(theta_nonorth, shade=True, color=colors[1])
        @savefig nonorth.png width=5in
        ax.axvline(0.5, color='k', label='True theta');

.. tabbed:: R

    .. code-block:: R

        > # R-code here
        > a=5

The regularization bias in the simple ML-approach is caused by the slow convergence of :math:`\hat{\theta}`

.. math::

    |\sqrt{n} (\hat{\theta} - \theta) | \rightarrow_{P} \infty

i.e. slower than :math:`1/\sqrt{n}`.
The driving factor is the bias in learning :math:`g`.
A Heuristic illustration is given by

.. math::

    \sqrt{n}(\hat{\theta} - \theta) = \underbrace{\left(\frac{1}{n} \sum_{i\in I} D_i^2\right)^{-1} \frac{1}{n} \sum_{i\in I} D_i U_i}_{=:a}
    +  \underbrace{\left(\frac{1}{n} \sum_{i\in I} D_i^2\right)^{-1} \frac{1}{n} \sum_{i\in I} D_i (g(X_i) - \hat{g}(X_i))}_{=:b}.

:math:`a` is approximately Gaussian under mild conditions.
However, :math:`b` (the regularization bias) diverges in general.

.. _bias_non_orth:

Overcoming Regularization Bias by Orthogonalization
+++++++++++++++++++++++++++++++++++++++++++++++++++

Directly partialling out the effect of :math:`X` from :math:`D` to obtain the orthogonalized regressor :math:`V = D - m(X)`.
We use the final estimate

.. math::

    \check{\theta} = \left(\frac{1}{n} \sum_{i\in I} \hat{V}_i D_i\right)^{-1} \frac{1}{n} \sum_{i\in I} \hat{V}_i (Y_i - \hat{g}(X_i)).

.. tabbed:: Python

    .. ipython:: python

        theta_orth_nosplit = np.zeros(n_rep)
        for i_rep in range(n_rep):
            (x, y, d) = data[i_rep]
            obj_dml_data = DoubleMLData.from_arrays(x, y, d)
            obj_dml_plr_orth_nosplit = DoubleMLPLR(obj_dml_data,
                                                   ml_g, ml_m,
                                                   n_folds=1,
                                                   score='IV-type')
            obj_dml_plr_orth_nosplit.fit()
            theta_orth_nosplit[i_rep] = obj_dml_plr_orth_nosplit.coef[0]

        ax = sns.kdeplot(theta_orth_nosplit, shade=True, color=colors[2])
        @savefig orth_nosplit.png width=5in
        ax.axvline(0.5, color='k', label='True theta');

.. tabbed:: R

    .. code-block:: R

        > # R-code here
        > a=5

If the nuisance models :math:`\hat{g}()` and :math:`\hat{m}()` are estimate on the whole dataset which is also used for obtaining
the final estimate :math:`\check{\theta}` another bias can be observed.

.. _bias_overfitting:

Sample Splitting to Remove Bias Induced by Overfitting
++++++++++++++++++++++++++++++++++++++++++++++++++++++

Using sample splitting, i.e., estimate the nuisance models :math:`\hat{g}()` and :math:`\hat{m}()` on one part of the
data (training data) and estimate :math:`\check{\theta}` on the other part of the data (test data) overcomes the bias
induced by overfitting. Cross-fitting performs well empirically.

.. tabbed:: Python

    .. ipython:: python

        theta_dml = np.zeros(n_rep)
        for i_rep in range(n_rep):
            (x, y, d) = data[i_rep]
            obj_dml_data = DoubleMLData.from_arrays(x, y, d)
            obj_dml_plr = DoubleMLPLR(obj_dml_data,
                                      ml_g, ml_m,
                                      n_folds=2,
                                      score='IV-type')
            obj_dml_plr.fit()
            theta_dml[i_rep] = obj_dml_plr.coef[0]

        ax = sns.kdeplot(theta_dml, shade=True, color=colors[3])
        @savefig orth.png width=5in
        ax.axvline(0.5, color='k', label='True theta');

.. tabbed:: R

    .. code-block:: R

        > # R-code here
        > a=5

Double/debiased machine learning
++++++++++++++++++++++++++++++++

To illustrate the benefits of the auxiliary prediction step (the DML) we write the error as

.. math::

    \sqrt{n}(\check{\theta} - \theta) = a^* + b^* + c^*

Chernozhukov et al. 2017 argues that:

The first term

.. math::

    a^* := (EV^2)^{-1} \frac{1}{\sqrt{n}} \sum_{i\in I} V_i U_i

will be asymptotically normally distributed.

The second term

.. math::

    b^* := (EV^2)^{-1} \frac{1}{\sqrt{n}} \sum_{i\in I} (\hat{m}(X_i) - m(X_i)) (\hat{g}(X_i) - g(X_i))

vanishes asymptotically for many data generating processes.

The third term :math:`c^*` vanishes in probability if sample splitting is applied.

.. tabbed:: Python

    .. ipython:: python

        ax = sns.kdeplot(theta_ols, shade=True)
        sns.kdeplot(theta_nonorth, shade=True, ax=ax);
        sns.kdeplot(theta_orth_nosplit, shade=True);
        sns.kdeplot(theta_dml, shade=True);
        labels = ['True Theta', 'OLS', 'Non-Orthogonal ML', 'Double ML (no Cross-Fitting)', 'Double ML with Cross-Fitting']
        ax.axvline(0.5, color='k', label='True theta');
        @savefig comparison.png width=5in
        ax.legend(labels);

.. tabbed:: R

    .. code-block:: R

        > # R-code here
        > a=5