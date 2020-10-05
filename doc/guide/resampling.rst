Sample-splitting, cross-fitting and repeated cross-fitting
----------------------------------------------------------

Sample-splitting and the application of cross-fitting is a central part of double/debiased machine learning (DML).
For all DML models
:class:`~doubleml.double_ml_plr.DoubleMLPLR`,
:class:`~doubleml.double_ml_pliv.DoubleMLPLIV`,
:class:`~doubleml.double_ml_irm.DoubleMLIRM`,
and :class:`~doubleml.double_ml_iivm.DoubleMLIIVM`
the specification is done via the parameters ``n_folds`` and ``n_rep_cross_fit``.
Advanced resampling techniques can be obtained via the boolean parameters
``draw_sample_splitting`` and ``apply_cross_fitting`` as well as the methods
``draw_sample_splitting()`` and ``set_sample_splitting()``.

As an example we consider a partially linear regression model (PLR)
implemented in :class:`~doubleml.double_ml_plr.DoubleMLPLR`.

.. tabbed:: Python

    .. ipython:: python

        import doubleml as dml
        import numpy as np
        from doubleml.datasets import make_plr_data
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.base import clone

        learner = RandomForestRegressor(max_depth=2, n_estimators=10)
        ml_g = clone(learner)
        ml_m = clone(learner)
        np.random.seed(123)
        data = make_plr_data()
        obj_dml_data = dml.DoubleMLData(data, 'y', 'd')

.. tabbed:: R

    .. jupyter-execute::

        library(DoubleML)
        library(mlr3)
        lgr::get_logger("mlr3")$set_threshold("warn")
        library(mlr3learners)
        library(data.table)

        learner = "regr.ranger"
        ml_g = learner
        ml_m = learner
        data("data_plr")
        data = data_plr
        obj_dml_data = DoubleMLData$new(data,
                                        y_col = "y",
                                        d_cols = "d")

.. _k-fold-cross-fitting:

Cross-fitting with :math:`K` folds
++++++++++++++++++++++++++++++++++

The default setting is ``n_folds = 5`` and ``n_rep_cross_fit = 1``, i.e.,
:math:`K=5` folds and no repeated cross-fitting.

.. tabbed:: Python

    .. ipython:: python

        dml_plr_obj =dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m, n_folds = 5, n_rep_cross_fit = 1)
        print(dml_plr_obj.n_folds)
        print(dml_plr_obj.n_rep_cross_fit)

.. tabbed:: R

    .. jupyter-execute::

        dml_plr_obj = DoubleMLPLR$new(obj_dml_data, ml_g, ml_m, n_folds = 5, n_rep_cross_fit = 1)
        print(dml_plr_obj$n_folds)
        print(dml_plr_obj$n_rep_cross_fit)

During the initialization of a DML model like :class:`~doubleml.double_ml_plr.DoubleMLPLR` a :math:`K`-fold random
partition :math:`(I_k)_{k=1}^{K}` of observation indices is generated.
The :math:`K`-fold random partition is stored in the ``smpls`` attribute of the DML model object.

.. TODO: add more detailed describtion of the ``smpls`` list. Or refer to the attribute description.

.. tabbed:: Python

    .. ipython:: python

        print(dml_plr_obj.smpls)

.. tabbed:: R

    .. jupyter-execute::

        dml_plr_obj$smpls

For each :math:`k \in [K] = \lbrace 1, \ldots, K]` the nuisance ML estimator

    .. math::

        \hat{\eta}_{0,k} = \hat{\eta}_{0,k}\big((W_i)_{i\not\in I_k}\big)

is based on the observations of all other :math:`k-1` folds.
The values of the two score function components
:math:`\psi_a(W_i; \hat{\eta}_0)` and :math:`\psi_b(W_i; \hat{\eta}_0))`
for each observation index :math:`i \in I_k` are computed and
stored in the attributes ``psi_a`` and ``psi_b``.

.. tabbed:: Python

    .. ipython:: python

        dml_plr_obj.fit()
        print(dml_plr_obj.psi_a[:5])
        print(dml_plr_obj.psi_b[:5])

.. tabbed:: R

    .. jupyter-execute::

        dml_plr_obj$fit()
        print(dml_plr_obj$.__enclos_env__$private$psi_a[1:5, ,1])
        print(dml_plr_obj$.__enclos_env__$private$psi_b[1:5, ,1])

Repeated cross-fitting with :math:`K` folds and :math:`M` repetition
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Repeated cross-fitting is obtained by choosing a value :math:`M>1` for the number of repetition ``n_rep_cross_fit``.
It results in :math:`M` random :math:`K`-fold partitions being drawn.

.. tabbed:: Python

    .. ipython:: python

        dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m, n_folds = 5, n_rep_cross_fit = 10)
        print(dml_plr_obj.n_folds)
        print(dml_plr_obj.n_rep_cross_fit)

.. tabbed:: R

    .. jupyter-execute::

        dml_plr_obj = DoubleMLPLR$new(obj_dml_data, ml_g, ml_m, n_folds = 5, n_rep_cross_fit = 10)
        print(dml_plr_obj$n_folds)
        print(dml_plr_obj$n_rep_cross_fit)

For each of the :math:`M` partitions, the nuisance ML models are estimated and score functions computed as described
in :ref:`k-fold-cross-fitting`.
The resulting values of the score functions are stored in 3-dimensional arrays ``psi_a`` and ``psi_b``, where the
row index corresponds the observation index :math:`i \in [N] = \lbrace 1, \ldots, N]`
and the column index to the partition :math:`m \in [M] = \lbrace 1, \ldots, M]`.
The third dimension refers to the treatment variable and becomes non-singleton in case of multiple treatment variables.

.. TODO: decide whether we always place hints with regards to the multiple treatment case or whether we always refer to the case of one treatment variable and the multiple treatment case is handled in one section of the documentation which is solely discussing the multiple treatment case.
.. Note that in case of multiple treatment variables the score functions are 3-dimensional arrays where the third dimension
.. refers to the different treatment variables.

.. tabbed:: Python

    .. ipython:: python

        dml_plr_obj.fit()
        print(dml_plr_obj.psi_a[:5, :, 0])
        print(dml_plr_obj.psi_b[:5, :, 0])

.. tabbed:: R

    .. jupyter-execute::

        dml_plr_obj$fit()
        print(dml_plr_obj$.__enclos_env__$private$psi_a[1:5, ,1])
        print(dml_plr_obj$.__enclos_env__$private$psi_b[1:5, ,1])

We estimate the causal parameter :math:`\tilde{\theta}_{0,m}` for each of the :math:`M` partitions with a DML
algorithm as described in :ref:`dml-algo`.
Standard errors are obtained as described in :ref:`se-confint`.
The aggregation of the estimates of the causal parameter and its standard errors is done using the median

    .. math::
        \tilde{\theta}_{0} &= \text{Median}\big((\tilde{\theta}_{0,m})_{m \in [M]}\big),

        \hat{\sigma} &= \sqrt{\text{Median}\big(\hat{\sigma}_m^2 - N (\tilde{\theta}_{0,m} - \tilde{\theta}_{0})^2\big)}.

The estimate of the causal parameter :math:`\tilde{\theta}_{0}` is stored in the ``coef`` attribute
and the asymptotic standard error :math:`\hat{\sigma}/\sqrt{N}` in ``se``.

.. tabbed:: Python

    .. ipython:: python

        print(dml_plr_obj.coef)
        print(dml_plr_obj.se)

.. tabbed:: R

    .. jupyter-execute::

        print(dml_plr_obj$coef)
        print(dml_plr_obj$se)

The parameter estimates :math:`(\tilde{\theta}_{0,m})_{m \in [M]}` and asymptotic standard errors
:math:`(\hat{\sigma}_m)_{m \in [M]}` for each of the :math:`M` partitions are stored in the attributes
``_all_coef`` and ``_all_se``, respectively.

.. tabbed:: Python

    .. ipython:: python

        print(dml_plr_obj._all_coef)
        print(dml_plr_obj._all_se)

.. tabbed:: R

    .. jupyter-execute::

        print(dml_plr_obj$.__enclos_env__$private$all_coef)
        print(dml_plr_obj$.__enclos_env__$private$all_se)

Externally provide a sample splitting / partition
+++++++++++++++++++++++++++++++++++++++++++++++++

All DML models allow a partition to be provided externally via the method ``set_sample_splitting()``.
For example we can use the K-Folds cross-validator of sklearn :py:class:`~sklearn.model_selection.KFold` in order to
generate a sample splitting and provide it to the DML model object.
Note that by setting ``draw_sample_splitting = False`` one can prevent that a partition is drawn during initialization
of the DML model object.
The following are equivalent.
In the first sample code, we use the standard interface and draw the sample-splitting with :math:`K=4` folds during
initialization of the :class:`~doubleml.double_ml_plr.DoubleMLPLR` object.

.. tabbed:: Python

    .. ipython:: python

        np.random.seed(314)
        dml_plr_obj_internal = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m, n_folds = 4)
        dml_plr_obj_internal.fit()
        print(dml_plr_obj_internal.summary)

.. tabbed:: R

    .. jupyter-execute::

        set.seed(314)
        dml_plr_obj_internal = DoubleMLPLR$new(obj_dml_data, ml_g, ml_m, n_folds = 4)
        dml_plr_obj_internal$fit()
        dml_plr_obj_internal$summary()

In the second sample code, we use the K-Folds cross-validator of sklearn :py:class:`~sklearn.model_selection.KFold`
and set the partition via the ``set_sample_splitting()`` method.

.. tabbed:: Python

    .. ipython:: python

        dml_plr_obj_external = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m, draw_sample_splitting = False)

        from sklearn.model_selection import KFold
        np.random.seed(314)
        kf = KFold(n_splits=4, shuffle=True)
        smpls = [[(train, test) for train, test in kf.split(obj_dml_data.x)]]

        dml_plr_obj_external.set_sample_splitting(smpls)
        dml_plr_obj_external.fit()
        print(dml_plr_obj_external.summary)

.. tabbed:: R

    .. jupyter-execute::

        dml_plr_obj_external = DoubleMLPLR$new(obj_dml_data, ml_g, ml_m, draw_sample_splitting = FALSE)

        set.seed(314)
        # set up a task and cross-validation resampling scheme in mlr3
        my_task = Task$new("help task", "regr", data)
        my_sampling = rsmp("cv", folds = 4)$instantiate(my_task)

        train_ids = lapply(1:4, function(x) my_sampling$train_set(x))
        test_ids = lapply(1:4, function(x) my_sampling$test_set(x))
        smpls = list(list(train_ids = train_ids, test_ids = test_ids))

        dml_plr_obj_external$set_samples(smpls)
        dml_plr_obj_external$fit()
        dml_plr_obj_external$summary()

Sample-splitting without cross-fitting
++++++++++++++++++++++++++++++++++++++

The boolean flag ``apply_cross_fitting`` allows to estimate DML models without applying cross-fitting.
It results in randomly splitting the sample into two parts.
The first half of the data is used for the estimation of the nuisance ML models and the second half for estimating the
causal parameter.
Note that cross-fitting performs well empirically and is recommended to remove bias induced by overfitting, see also
:ref:`bias_overfitting`.

.. tabbed:: Python

    .. ipython:: python

        dml_plr_obj_external = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m,
                                               n_folds = 2, apply_cross_fitting = False)
        dml_plr_obj_external.fit()
        print(dml_plr_obj_external.summary)
        print(dml_plr_obj_external.n_obs)
        print(dml_plr_obj_external.psi.shape)

.. tabbed:: R

    .. jupyter-execute::

        dml_plr_obj_external = DoubleMLPLR$new(obj_dml_data, ml_g, ml_m,
                                               n_folds = 2, apply_cross_fitting = FALSE)
        dml_plr_obj_external$fit()
        dml_plr_obj_external$summary()
        print(dml_plr_obj_external$n_obs)
        print(dim(dml_plr_obj_external$.__enclos_env__$private$psi))
Note, that in order to split data unevenly into train and test the interface to externally set the sample splitting
via ``set_sample_splitting()`` needs to be applied, like for example:

.. tabbed:: Python

    .. ipython:: python

        dml_plr_obj_external = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m,
                                               n_folds = 2, apply_cross_fitting = False, draw_sample_splitting = False)

        from sklearn.model_selection import train_test_split
        smpls = train_test_split(np.arange(obj_dml_data.n_obs), train_size=0.8)
        smpls = [np.sort(x) for x in smpls]  # only sorted indices are supported
        dml_plr_obj_external.set_sample_splitting([[smpls]])

        dml_plr_obj_external.fit()
        print(dml_plr_obj_external.summary)
        print(dml_plr_obj_external.n_obs)
        print(dml_plr_obj_external.psi.shape)

.. tabbed:: R

    .. jupyter-execute::

        dml_plr_obj_external = DoubleMLPLR$new(obj_dml_data, ml_g, ml_m,
                                                n_folds = 2, apply_cross_fitting = FALSE, draw_sample_splitting = FALSE)

        set.seed(314)
        # set up a task and cross-validation resampling scheme in mlr3
        my_task = Task$new("help task", "regr", data)
        my_sampling = rsmp("holdout", ratio = 0.8)$instantiate(my_task)

        train_ids = list(my_sampling$train_set(1))
        test_ids = list(my_sampling$test_set(1))
        smpls = list(list(train_ids = train_ids, test_ids = test_ids))

        dml_plr_obj_external$set_samples(smpls)
        dml_plr_obj_external$fit()
        dml_plr_obj_external$summary()
        print(dml_plr_obj_external$n_obs)
        print(dim(dml_plr_obj_external$.__enclos_env__$private$psi))
