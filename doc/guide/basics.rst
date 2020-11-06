.. _basics:

The basics of double/debiased machine learning
----------------------------------------------

In the following we provide a brief summary of and motivation to double machine learning methods and show how the
corresponding methods provided by the :ref:`DoubleML <doubleml_package>` package can be applied.
For details we refer to Chernozhukov et al. (2018).

.. Add references to the vignette here when it is ready.

Data generating process
+++++++++++++++++++++++

We consider the following partially linear model

.. math::

        d_i = m_0(x_i) + v_i, & &v_i \sim \mathcal{N}(0,1),

        y_i = \theta d_i + g_0(x_i) + \zeta_i, & &\zeta_i \sim \mathcal{N}(0,1),


with covariates :math:`x_i \sim \mathcal{N}(0, \Sigma)`, where  :math:`\Sigma` is a matrix with entries
:math:`\Sigma_{kj} = 0.7^{|j-k|}`.
The true parameter :math:`\theta` is set to :math:`0.5`.

The nuisance functions are given by

.. math::

    m_0(x_i) &= \frac{1}{4} x_{i,1} + \frac{\exp(x_{i,3})}{1+\exp(x_{i,3})},

    g_0(X) &= \frac{\exp(x_{i,1})}{1+\exp(x_{i,1})} + \frac{1}{4} x_{i,3}.


.. note::
    - In Python the data can be generated with :py:func:`doubleml.datasets.make_plr_CCDDHNR2018`.
    - In R the data can be generated with `DoubleML::make_plr_CCDDHNR2018()`.

.. tabbed:: Python

    .. ipython:: python

        import numpy as np
        from doubleml.datasets import make_plr_CCDDHNR2018

        np.random.seed(1234)
        n_rep = 100
        n_obs = 500
        n_vars = 20
        alpha = 0.5

        data = list()

        for i_rep in range(n_rep):
            (x, y, d) = make_plr_CCDDHNR2018(alpha=alpha, n_obs=n_obs, dim_x=n_vars, return_type='array')
            data.append((x, y, d))

.. tabbed:: R

    .. jupyter-execute::

        library(DoubleML)
        set.seed(1234)
        n_rep = 100
        n_obs = 500
        n_vars = 20
        alpha = 0.5

        data = list()
        for (i_rep in seq_len(n_rep)) {
            data[[i_rep]] = make_plr_CCDDHNR2018(alpha=alpha, n_obs=n_obs, dim_x=n_vars,
                                                  return_type="data.frame")
        }


OLS estimation
++++++++++++++

A naive OLS regression of :math:`Y` on :math:`D` produces a significant bias.

.. tabbed:: Python

    .. The following block makes sure that the seaborn graphics are rendered appropriately but does not need to be shown

    .. ipython:: python
        :suppress:

        import seaborn as sns
        sns.set()

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

        # to speed up the illustration we hard-code the simulation results
        theta_ols = np.array([0.74327549, 0.71934891, 0.7071162 , 0.66889543, 0.7321444 , 0.69223437, 0.68652905, 0.68488856, 0.72617801, 0.77559064, 0.71563239, 0.70957142, 0.72750812, 0.61424786, 0.71949073, 0.76285484, 0.7496472 , 0.69688757, 0.6478375 , 0.69363074, 0.60957089, 0.69387009, 0.63558263, 0.74582498, 0.68074102, 0.66290947, 0.70199172, 0.67168635, 0.6949646 , 0.67395231, 0.6863304 , 0.72837954, 0.63418985, 0.624109  , 0.74254538, 0.66124323, 0.69945743, 0.74743667, 0.67949935, 0.70447876, 0.68997626, 0.6485608 , 0.6996691 , 0.64571962, 0.689712  , 0.72131247, 0.75407754, 0.70783852, 0.66223883, 0.74018891, 0.71145533, 0.69873155, 0.7445769 , 0.74689479, 0.73542415, 0.67113299, 0.76903052, 0.70375282, 0.70996563, 0.70438228, 0.74164684, 0.74533407, 0.66750142, 0.66591911, 0.68102801, 0.68273323, 0.68804608, 0.65742228, 0.73445098, 0.73037019, 0.70510367, 0.66405038, 0.7135943 , 0.68373932, 0.69066059, 0.72144004, 0.70095231, 0.73540764, 0.65717624, 0.72036667, 0.68831806, 0.68206625, 0.6739611 , 0.6508514 , 0.68228283, 0.70360066, 0.69994817, 0.6896916 , 0.7008032 , 0.65414898, 0.63797163, 0.68304265, 0.719207  , 0.71895005, 0.68102457, 0.72023145, 0.65410707, 0.73580482, 0.758165  , 0.76284332])

        # to run the full simulation uncomment the following line to fit the model for every dataset and not just for the first dataset
        #for i_rep in range(n_rep):
        for i_rep in range(1):
            (x, y, d) = data[i_rep]
            this_theta = est_ols(y, d)
            # we show that the loaded result matches the just computed
            print(np.abs(theta_ols[i_rep] - this_theta))
            theta_ols[i_rep] = this_theta

        ax = sns.kdeplot(theta_ols, shade=True, color=colors[0])
        @savefig ols.png width=5in
        ax.axvline(0.5, color='k', label='True theta');

.. tabbed:: R

    .. jupyter-execute::

        library(ggplot2)

        est_ols = function(df) {
            ols = stats::lm(y ~ -1 +., df)
            theta = coef(ols)["d"]
            return(theta)
        }

        theta_ols = rep(0, n_rep)
        for (i_rep in seq_len(n_rep)) {
            df = data[[i_rep]]
            theta_ols[i_rep] = est_ols(df)
        }

        g_ols = ggplot(data.frame(theta_ols), aes(x = theta_ols)) +
                    geom_density(fill = "dark blue", alpha = 0.3, color = "dark blue") +
                    geom_vline(aes(xintercept = alpha), col = "black")
        g_ols


Regularization bias in simple ML-approaches
+++++++++++++++++++++++++++++++++++++++++++

A simple ML approach is given by randomly splitting the sample into two parts.
On the auxiliary sample indexed by :math:`i \in I^C` the nuisance function :math:`g_0(X)` is estimated with an ML method.
Given the estimate :math:`\hat{g}_0(X)`, the final estimate of :math:`\theta` is obtained as (:math:`n=N/2`) using the
other half of observations indexed with :math:`i \in I`


.. math::

    \hat{\theta} = \left(\frac{1}{n} \sum_{i\in I} D_i^2\right)^{-1} \frac{1}{n} \sum_{i\in I} D_i (Y_i - \hat{g}_0(X_i)).

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
        import numpy as np
        np.random.seed(1111)

        learner = RandomForestRegressor(n_estimators=500)
        ml_m = clone(learner)
        ml_g = clone(learner)

        # to speed up the illustration we hard-code the simulation results
        theta_nonorth = np.array([0.29366923, 0.3185203 , 0.37533048, 0.37088366, 0.19803773, 0.30828113, 0.34663329, 0.27895468, 0.29746076, 0.24940245, 0.29157326, 0.38012658, 0.35148578, 0.26273756, 0.33198856, 0.35471521, 0.4032286 , 0.2972334 , 0.32185157, 0.23044533, 0.3186483 , 0.32460869, 0.22068749, 0.3613337 , 0.35556874, 0.28618608, 0.29061311, 0.16806013, 0.34984153, 0.2682979 , 0.15792403, 0.26568878, 0.19371346, 0.24222099, 0.28596736, 0.26820864, 0.25735315, 0.28764038, 0.27602423, 0.32650334, 0.25556837, 0.26330747, 0.31135055, 0.18424877, 0.25083002, 0.24005185, 0.29130529, 0.32371353, 0.30196971, 0.23685769, 0.33663465, 0.35461096, 0.33337579, 0.36119784, 0.34360896, 0.25420008, 0.18020323, 0.36871964, 0.35384154, 0.33841536, 0.27163847, 0.25221933, 0.26272111, 0.25356449, 0.267646  , 0.19191615, 0.28093068, 0.27512042, 0.36157435, 0.17145189, 0.12915806, 0.24374555, 0.27688143, 0.12256028, 0.43368743, 0.23872294, 0.34576773, 0.26492525, 0.34938815, 0.29693057, 0.27551534, 0.3528473 , 0.33108614, 0.1452295 , 0.22088902, 0.24975281, 0.32121537, 0.34396557, 0.40426961, 0.3513969 , 0.24562922, 0.22752121, 0.37189554, 0.18458102, 0.30684071, 0.34418113, 0.30683124, 0.15732466, 0.4439994 , 0.22972917])

        # to run the full simulation uncomment the following line to fit the model for every dataset and not just for the first dataset
        #for i_rep in range(n_rep):
        for i_rep in range(1):
            (x, y, d) = data[i_rep]
            obj_dml_data = DoubleMLData.from_arrays(x, y, d)
            obj_dml_plr_nonorth = DoubleMLPLR(obj_dml_data,
                                              ml_m, ml_g,
                                              n_folds=2,
                                              apply_cross_fitting=False,
                                              score=non_orth_score)
            obj_dml_plr_nonorth.fit()
            this_theta = obj_dml_plr_nonorth.coef[0]
            # we show that the loaded result matches the just computed
            print(np.abs(theta_nonorth[i_rep] - this_theta))
            theta_nonorth[i_rep] = this_theta

        ax = sns.kdeplot(theta_nonorth, shade=True, color=colors[1])
        @savefig nonorth.png width=5in
        ax.axvline(0.5, color='k', label='True theta');

.. tabbed:: R

    .. jupyter-execute::

        non_orth_score = function(y, d, g_hat, m_hat, smpls) {
         u_hat = y - g_hat
         psi_a = -1*d*d
         psi_b = d*u_hat
         psis = list(psi_a = psi_a, psi_b = psi_b)
         return(psis)
        }


    .. jupyter-execute::

        # not yet implemented in R #
        library(mlr3)
        library(mlr3learners)
        library(data.table)
        lgr::get_logger("mlr3")$set_threshold("warn")

        learner = lrn("regr.ranger", num.trees = 500)
        ml_m = learner$clone()
        ml_g = learner$clone()
        theta_nonorth = rep(0, n_rep)

        for (i_rep in seq_len(n_rep)) {
            df = data[[i_rep]]
            obj_dml_data = double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")
            obj_dml_plr_nonorth = DoubleMLPLR$new(obj_dml_data,
                                                   ml_g, ml_m,
                                                   n_folds=2,
                                                   score=non_orth_score,
                                                   apply_cross_fitting=FALSE)
            #obj_dml_plr_nonorth$fit()
            theta_nonorth[i_rep] = obj_dml_plr_nonorth$coef
        }
        g_nonorth = ggplot(data.frame(theta_nonorth), aes(x = theta_nonorth)) +
                        geom_density(fill = "dark orange", alpha = 0.3, color = "dark orange") +
                        geom_vline(aes(xintercept = alpha), col = "black")
        g_nonorth


The regularization bias in the simple ML-approach is caused by the slow convergence of :math:`\hat{\theta}`

.. math::

    |\sqrt{n} (\hat{\theta} - \theta) | \rightarrow_{P} \infty

i.e. slower than :math:`1/\sqrt{n}`.
The driving factor is the bias in learning :math:`g`.
A Heuristic illustration is given by

.. math::

    \sqrt{n}(\hat{\theta} - \theta) = \underbrace{\left(\frac{1}{n} \sum_{i\in I} D_i^2\right)^{-1} \frac{1}{n} \sum_{i\in I} D_i U_i}_{=:a}
    +  \underbrace{\left(\frac{1}{n} \sum_{i\in I} D_i^2\right)^{-1} \frac{1}{n} \sum_{i\in I} D_i (g_0(X_i) - \hat{g}_0(X_i))}_{=:b}.

:math:`a` is approximately Gaussian under mild conditions.
However, :math:`b` (the regularization bias) diverges in general.

.. _bias_non_orth:

Overcoming regularization bias by orthogonalization
+++++++++++++++++++++++++++++++++++++++++++++++++++

To overcome the regularization bias we are directly partialling out the effect of :math:`X` from :math:`D` to obtain the
orthogonalized regressor :math:`V = D - m(X)`. We then use the final estimate

.. math::

    \check{\theta} = \left(\frac{1}{n} \sum_{i\in I} \hat{V}_i D_i\right)^{-1} \frac{1}{n} \sum_{i\in I} \hat{V}_i (Y_i - \hat{g}_0(X_i)).

.. tabbed:: Python

    .. ipython:: python

        import numpy as np
        np.random.seed(2222)

        # to speed up the illustration we hard-code the simulation results
        theta_orth_nosplit = np.array([0.21267246, 0.19035703, 0.17553032, 0.18811128, 0.19009917, 0.18871992, 0.20287762, 0.18080227, 0.20571531, 0.23544614, 0.20168225, 0.20249789, 0.21004924, 0.15928406, 0.18548533, 0.18651856, 0.19766646, 0.17434387, 0.19763059, 0.19645366, 0.18455094, 0.16241166, 0.16966572, 0.18960055, 0.1810025 , 0.1832846 , 0.19458859, 0.1791934 , 0.17675649, 0.19951372, 0.2011414 , 0.22412837, 0.15889931, 0.15687732, 0.18535516, 0.16488848, 0.18261685, 0.20954582, 0.19590937, 0.18625687, 0.17188457, 0.1858343 , 0.18879836, 0.14296883, 0.20590783, 0.18350397, 0.19686679, 0.18653557, 0.20284635, 0.22071435, 0.18611925, 0.18737602, 0.20448496, 0.19888918, 0.21829413, 0.1691084 , 0.19997557, 0.21021366, 0.176807  , 0.17291738, 0.20451048, 0.19983547, 0.21683657, 0.18955314, 0.21238901, 0.16822338, 0.17381694, 0.192073  , 0.21816915, 0.19955613, 0.17032554, 0.17568031, 0.1776993 , 0.18832984, 0.19328225, 0.20206436, 0.1895523 , 0.19932858, 0.18523962, 0.21029524, 0.18919887, 0.19406502, 0.18869902, 0.17773589, 0.16403292, 0.19089166, 0.1959437 , 0.19981805, 0.19809157, 0.19459026, 0.16202348, 0.17559719, 0.22848809, 0.19218543, 0.19113699, 0.19115575, 0.19039321, 0.19983057, 0.22975582, 0.20277958])

        # to run the full simulation uncomment the following line to fit the model for every dataset and not just for the first dataset
        #for i_rep in range(n_rep):
        for i_rep in range(1):
            (x, y, d) = data[i_rep]
            obj_dml_data = DoubleMLData.from_arrays(x, y, d)
            obj_dml_plr_orth_nosplit = DoubleMLPLR(obj_dml_data,
                                                   ml_g, ml_m,
                                                   n_folds=1,
                                                   score='IV-type',
                                                   apply_cross_fitting=False)
            obj_dml_plr_orth_nosplit.fit()
            this_theta = obj_dml_plr_orth_nosplit.coef[0]
            # we show that the loaded result matches the just computed
            print(np.abs(theta_orth_nosplit[i_rep] - this_theta))
            theta_orth_nosplit[i_rep] = this_theta

        ax = sns.kdeplot(theta_orth_nosplit, shade=True, color=colors[2])
        @savefig orth_nosplit.png width=5in
        ax.axvline(0.5, color='k', label='True theta');

.. tabbed:: R

    .. jupyter-execute::

        library(data.table)
        lgr::get_logger("mlr3")$set_threshold("warn")

        theta_orth_nosplit = rep(0, n_rep)

        for (i_rep in seq_len(n_rep)) {
            df = data[[i_rep]]
            obj_dml_data = double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")
            obj_dml_plr_orth_nosplit = DoubleMLPLR$new(obj_dml_data,
                                                   ml_g, ml_m,
                                                   n_folds=1,
                                                   score='IV-type',
                                                   apply_cross_fitting=FALSE)
            #obj_dml_plr_orth_nosplit$fit()
            theta_orth_nosplit[i_rep] = obj_dml_plr_orth_nosplit$coef
        }
        g_nosplit = ggplot(data.frame(theta_orth_nosplit), aes(x = theta_orth_nosplit)) +
                    geom_density(fill = "dark green", alpha = 0.3, color = "dark green") +
                    geom_vline(aes(xintercept = alpha), col = "black")
        g_nosplit


If the nuisance models :math:`\hat{g}_0()` and :math:`\hat{m}()` are estimate on the whole dataset which is also used for obtaining
the final estimate :math:`\check{\theta}` another bias can be observed.

.. _bias_overfitting:

Sample splitting to remove bias induced by overfitting
++++++++++++++++++++++++++++++++++++++++++++++++++++++

Using sample splitting, i.e., estimate the nuisance models :math:`\hat{g}_0()` and :math:`\hat{m}()` on one part of the
data (training data) and estimate :math:`\check{\theta}` on the other part of the data (test data) overcomes the bias
induced by overfitting. Cross-fitting performs well empirically.

.. tabbed:: Python

    .. ipython:: python

        import numpy as np
        np.random.seed(3333)

        # to speed up the illustration we hard-code the simulation results
        theta_dml = np.array([0.54370516, 0.55369544, 0.50097982, 0.52045338, 0.53547201, 0.46795147, 0.54512537, 0.4741297 , 0.58580663, 0.64865909, 0.5201819 , 0.54087818, 0.54222501, 0.46950931, 0.49654939, 0.50097532, 0.55123242, 0.50490227, 0.55370679, 0.53199869, 0.48967937, 0.43577411, 0.46523043, 0.5493953 , 0.46254555, 0.50608273, 0.55363785, 0.51922619, 0.52676764, 0.4838423 , 0.51066561, 0.58334809, 0.46378467, 0.4642138 , 0.44336647, 0.45042542, 0.52993666, 0.58578501, 0.48790803, 0.54744297, 0.43473949, 0.43933014, 0.50520622, 0.44414833, 0.5980775 , 0.48150111, 0.54326984, 0.56599116, 0.57365272, 0.59776555, 0.49154794, 0.55213332, 0.52850586, 0.52698463, 0.59896053, 0.4642551 , 0.62334582, 0.55125113, 0.51414727, 0.51910541, 0.54772369, 0.53309515, 0.52693028, 0.48280077, 0.59887037, 0.44053728, 0.53403916, 0.50420431, 0.5990181 , 0.51283072, 0.51651118, 0.4873045 , 0.49561432, 0.54101811, 0.5434123 , 0.60495487, 0.58368309, 0.51277121, 0.5162138 , 0.54098573, 0.50907415, 0.49830674, 0.50543125, 0.48862414, 0.44121645, 0.50641711, 0.57551834, 0.59563729, 0.49882687, 0.51580948, 0.48474833, 0.47058381, 0.62205853, 0.45802027, 0.56951467, 0.52920686, 0.50473622, 0.49641629, 0.59351419, 0.55661256])

        # to run the full simulation uncomment the following line to fit the model for every dataset and not just for the first dataset
        #for i_rep in range(n_rep):
        for i_rep in range(1):
            (x, y, d) = data[i_rep]
            obj_dml_data = DoubleMLData.from_arrays(x, y, d)
            obj_dml_plr = DoubleMLPLR(obj_dml_data,
                                      ml_g, ml_m,
                                      n_folds=2,
                                      score='IV-type')
            obj_dml_plr.fit()
            this_theta = obj_dml_plr.coef[0]
            # we show that the loaded result matches the just computed
            print(np.abs(theta_dml[i_rep] - this_theta))
            theta_dml[i_rep] = this_theta

        ax = sns.kdeplot(theta_dml, shade=True, color=colors[3])
        @savefig orth.png width=5in
        ax.axvline(0.5, color='k', label='True theta');

.. tabbed:: R

    .. jupyter-execute::

        theta_dml = rep(0, n_rep)
        for (i_rep in seq_len(n_rep)) {
            df = data[[i_rep]]
            obj_dml_data = double_ml_data_from_data_frame(df, y_col = "y", d_cols = "d")
            obj_dml_plr = DoubleMLPLR$new(obj_dml_data,
                                      ml_g, ml_m,
                                      n_folds=2,
                                      score='IV-type')
            #obj_dml_plr$fit()
            theta_dml[i_rep] = obj_dml_plr$coef
        }

        g_dml = ggplot(data.frame(theta_dml), aes(x = theta_dml)) +
                    geom_density(fill = "dark red", alpha = 0.3, color = "dark red") +
                    geom_vline(aes(xintercept = alpha), col = "black")
        g_dml


Double/debiased machine learning
++++++++++++++++++++++++++++++++

To illustrate the benefits of the auxiliary prediction step (the DML) we write the error as

.. math::

    \sqrt{n}(\check{\theta} - \theta) = a^* + b^* + c^*

Chernozhukov et al. (2018) argues that:

The first term

.. math::

    a^* := (EV^2)^{-1} \frac{1}{\sqrt{n}} \sum_{i\in I} V_i U_i

will be asymptotically normally distributed.

The second term

.. math::

    b^* := (EV^2)^{-1} \frac{1}{\sqrt{n}} \sum_{i\in I} (\hat{m}(X_i) - m(X_i)) (\hat{g}_0(X_i) - g_0(X_i))

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

    .. jupyter-execute::

        g_all = ggplot(data.frame(theta_ols, theta_nonorth, theta_orth_nosplit, theta_dml)) +
                    geom_density(aes(x = theta_ols), fill = "dark blue", alpha = 0.3, color = "dark blue") +
                    geom_density(aes(x = theta_nonorth), fill = "dark orange", alpha = 0.3, color = "dark orange") +
                    geom_density(aes(x = theta_orth_nosplit), fill = "dark green", alpha = 0.3, color = "dark green") +
                    geom_density(aes(x = theta_dml), fill = "dark red", alpha = 0.3, color = "dark red") +
                    geom_vline(aes(xintercept = alpha), col = "black")
        g_all


References
++++++++++

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J. (2018), Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21: C1-C68. doi:`10.1111/ectj.12097 <https://doi.org/10.1111/ectj.12097>`_.