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

.. tabs::

    .. code-tab:: py

        >>> import doubleml as dml
        >>> import numpy as np
        >>> from doubleml.datasets import make_plr_data
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from sklearn.base import clone

        >>> learner = RandomForestRegressor(max_depth=2, n_estimators=10)
        >>> ml_g = clone(learner)
        >>> ml_m = clone(learner)
        >>> np.random.seed(123)
        >>> data = make_plr_data()
        >>> obj_dml_data = dml.DoubleMLData(data, 'y', 'd')

    .. code-tab:: r R

        > # R-code here
        > a=5

.. _k-fold-cross-fitting:

Cross-fitting with :math:`K` folds
++++++++++++++++++++++++++++++++++

The default setting is ``n_folds = 5`` and ``n_rep_cross_fit = 1``, i.e.,
:math:`K=5` folds and no repeated cross-fitting.

.. tabs::

    .. code-tab:: py

        >>> dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m, n_folds = 5, n_rep_cross_fit = 1)
        >>> print(dml_plr_obj.n_folds)
        5
        >>> print(dml_plr_obj.n_rep_cross_fit)
        1

    .. code-tab:: r R

        > # R-code here
        > a=5

During the initialization of a DML model like :class:`~doubleml.double_ml_plr.DoubleMLPLR` a :math:`K`-fold random
partition :math:`(I_k)_{k=1}^{K}` of observation indices is generated.
The :math:`K`-fold random partition is stored in the ``smpls`` attribute of the DML model object.

.. TODO: add more detailed describtion of the ``smpls`` list. Or refer to the attribute description.

.. tabs::

    .. code-tab:: py

        >>> print(dml_plr_obj.smpls)
        [[(array([ 1,  3,  4,  5,  6,  9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
               22, 24, 25, 27, 29, 30, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 43,
               45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62,
               64, 65, 66, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84,
               85, 86, 87, 88, 89, 90, 94, 95, 96, 97, 98, 99]), array([ 0,  2,  7,  8, 12, 23, 26, 28, 34, 39, 44, 61, 63, 68, 69, 71, 82,
               91, 92, 93])), (array([ 0,  1,  2,  3,  5,  6,  7,  8,  9, 10, 11, 12, 13, 16, 17, 18, 23,
               26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
               44, 45, 47, 48, 49, 50, 52, 53, 54, 55, 56, 59, 60, 61, 63, 64, 65,
               67, 68, 69, 70, 71, 72, 73, 75, 76, 78, 79, 80, 81, 82, 83, 84, 85,
               86, 87, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99]), array([ 4, 14, 15, 19, 20, 21, 22, 24, 25, 30, 46, 51, 57, 58, 62, 66, 74,
               77, 88, 96])), (array([ 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 18, 19,
               20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38,
               39, 40, 41, 44, 45, 46, 47, 48, 51, 52, 53, 55, 56, 57, 58, 59, 61,
               62, 63, 64, 66, 67, 68, 69, 71, 74, 75, 76, 77, 78, 79, 80, 81, 82,
               83, 84, 88, 89, 90, 91, 92, 93, 95, 96, 98, 99]), array([ 3, 16, 17, 31, 32, 42, 43, 49, 50, 54, 60, 65, 70, 72, 73, 85, 86,
               87, 94, 97])), (array([ 0,  1,  2,  3,  4,  6,  7,  8, 10, 12, 14, 15, 16, 17, 18, 19, 20,
               21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 38, 39, 40,
               41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 54, 57, 58, 59, 60, 61, 62,
               63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 77, 78, 79, 80, 82, 85,
               86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98]), array([ 5,  9, 11, 13, 35, 36, 37, 45, 48, 53, 55, 56, 67, 75, 76, 81, 83,
               84, 95, 99])), (array([ 0,  2,  3,  4,  5,  7,  8,  9, 11, 12, 13, 14, 15, 16, 17, 19, 20,
               21, 22, 23, 24, 25, 26, 28, 30, 31, 32, 34, 35, 36, 37, 39, 42, 43,
               44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63,
               65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 81, 82, 83, 84,
               85, 86, 87, 88, 91, 92, 93, 94, 95, 96, 97, 99]), array([ 1,  6, 10, 18, 27, 29, 33, 38, 40, 41, 47, 52, 59, 64, 78, 79, 80,
               89, 90, 98]))]]

    .. code-tab:: r R

        > # R-code here
        > a=5

For each :math:`k \in [K] = \lbrace 1, \ldots, K]` the nuisance ML estimator

    .. math::

        \hat{\eta}_{0,k} = \hat{\eta}_{0,k}\big((W_i)_{i\not\in I_k}\big)

is based on the observations of all other :math:`k-1` folds.
The values of the two score function components
:math:`\psi_a(W_i; \hat{\eta}_0)` and :math:`\psi_b(W_i; \hat{\eta}_0))`
for each observation index :math:`i \in I_k` are computed and
stored in the attributes ``psi_a`` and ``psi_b``.

.. tabs::

    .. code-tab:: py

        >>> dml_plr_obj.fit()
        >>> print(dml_plr_obj.psi_a[:5])
        [[[-1.12677759e+00]]
         [[-9.50079642e-05]]
         [[-6.83377006e-02]]
         [[-3.59575747e-03]]
         [[-3.23296763e-01]]]
        >>> print(dml_plr_obj.psi_b[:5])
        [[[ 2.53537374]]
         [[ 0.0103673 ]]
         [[ 0.32162043]]
         [[ 0.04712103]]
         [[-0.69486749]]]

    .. code-tab:: r R

        > # R-code here
        > a=5

Repeated cross-fitting with :math:`K` folds and :math:`M` repetition
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Repeated cross-fitting is obtained by choosing a value :math:`M>1` for the number of repetition ``n_rep_cross_fit``.
It results in :math:`M` random :math:`K`-fold partitions being drawn.

.. tabs::

    .. code-tab:: py

        >>> dml_plr_obj = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m, n_folds = 5, n_rep_cross_fit = 10)
        >>> print(dml_plr_obj.n_folds)
        5
        >>> print(dml_plr_obj.n_rep_cross_fit)
        10

    .. code-tab:: r R

        > # R-code here
        > a=5

For each of the :math:`M` partitions, the nuisance ML models are estimated and score functions computed as described
in :ref:`k-fold-cross-fitting`.
The resulting values of the score functions are stored in 3-dimensional arrays ``psi_a`` and ``psi_b``, where the
row index corresponds the observation index :math:`i \in [N] = \lbrace 1, \ldots, N]`
and the column index to the partition :math:`m \in [M] = \lbrace 1, \ldots, M]`.
The third dimension refers to the treatment variable and becomes non-singleton in case of multiple treatment variables.

.. TODO: decide whether we always place hints with regards to the multiple treatment case or whether we always refer to the case of one treatment variable and the multiple treatment case is handled in one section of the documentation which is solely discussing the multiple treatment case.
.. Note that in case of multiple treatment variables the score functions are 3-dimensional arrays where the third dimension
.. refers to the different treatment variables.

.. tabs::

    .. code-tab:: py

        >>> dml_plr_obj.fit()
        >>> print(dml_plr_obj.psi_a[:5, :, 0])
        [[-1.58117307e+00 -1.14270400e+00 -8.86667526e-01 -7.34057694e-01
          -1.43320815e+00 -1.07295264e+00 -1.30977124e+00 -1.44455311e+00
          -1.25539854e+00 -1.28815908e+00]
         [-4.09078076e-02 -1.31382008e-01 -3.29166650e-02 -4.60378618e-04
          -7.04353533e-02 -9.42727164e-03 -7.88224772e-02 -2.36415316e-05
          -5.36496833e-03 -7.34043855e-03]
         [-4.19133186e-02 -1.00583249e-01 -5.26847379e-02 -1.69413957e-01
          -1.01654418e-01 -4.70086352e-02 -2.00882707e-01 -1.96646299e-01
          -1.48706039e-01 -3.94486719e-01]
         [-4.91131361e-03 -5.82298321e-03 -1.77675792e-02 -1.41141593e-04
          -7.11474215e-02 -5.11295847e-02 -4.00285092e-02 -5.09847759e-04
          -2.43051991e-02 -9.07581346e-02]
         [-2.74922633e-01 -4.53375739e-01 -5.87237865e-01 -5.10334865e-01
          -7.46200085e-01 -2.94879426e-01 -4.84186765e-01 -1.78503686e-01
          -1.09734503e+00 -4.84673033e-01]]
        >>> print(dml_plr_obj.psi_b[:5, :, 0])
        [[ 2.93539733  2.54876087  2.19322027  1.55493841  3.0683359   2.09100629
           2.76647112  2.76202109  2.27901084  2.49140458]
         [ 0.21499061  0.33397491  0.18244175 -0.01904318  0.28883258  0.11007686
           0.31361941  0.00859476 -0.09295213 -0.0754732 ]
         [ 0.20794882  0.47312624  0.22546044  0.44810997  0.4301562   0.34652429
           0.21444254  0.55094695  0.50870165  0.4884872 ]
         [ 0.05010272 -0.05743753 -0.12633266  0.01042029 -0.15967769 -0.11504402
          -0.12715838 -0.01521333  0.08715671 -0.17715329]
         [-0.90111543 -0.17925542 -0.37860714 -0.35840033 -0.35644447 -0.63278205
           0.01698057 -0.09038896 -0.27933253 -0.68081414]]

    .. code-tab:: r R

        > # R-code here
        > a=5

We estimate the causal parameter :math:`\tilde{\theta}_{0,m}` for each of the :math:`M` partitions with a DML
algorithm as described in :ref:`dml-algo`.
Standard errors are obtained as described in :ref:`se-confint`.
The aggregation of the estimates of the causal parameter and its standard errors is done using the median

    .. math::
        \tilde{\theta}_{0} &= \text{Median}\big((\tilde{\theta}_{0,m})_{m \in [M]}\big),

        \hat{\sigma} &= \sqrt{\text{Median}\big(\hat{\sigma}_m^2 - N (\tilde{\theta}_{0,m} - \tilde{\theta}_{0})^2\big)}.

The estimate of the causal parameter :math:`\tilde{\theta}_{0}` is stored in the ``coef`` attribute
and the asymptotic standard error :math:`\hat{\sigma}/\sqrt{N}` in ``se``.

.. tabs::

    .. code-tab:: py

        >>> print(dml_plr_obj.coef)
        [0.47487797]
        >>> print(dml_plr_obj.se)
        [0.13678215]

    .. code-tab:: r R

        > # R-code here
        > a=5

The parameter estimates :math:`(\tilde{\theta}_{0,m})_{m \in [M]}` and asymptotic standard errors
:math:`(\hat{\sigma}_m)_{m \in [M]}` for each of the :math:`M` partitions are stored in the attributes
``_all_coef`` and ``_all_se``, respectively.

.. tabs::

    .. code-tab:: py

        >>> print(dml_plr_obj._all_coef)
        [[0.39722212 0.47252568 0.46934501 0.48501473 0.482182   0.47723026
          0.43631832 0.49540176 0.44698793 0.49289055]]
        >>> print(dml_plr_obj._all_se)
        [[0.13802702 0.13610014 0.15692213 0.1378541  0.14287733 0.15353906
          0.13325949 0.13940565 0.13699459 0.13403344]]

    .. code-tab:: r R

        > # R-code here
        > a=5

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

.. tabs::

    .. code-tab:: py

        >>> np.random.seed(314)
        >>> dml_plr_obj_internal = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m, n_folds = 4)
        >>> dml_plr_obj_internal.fit()
        >>> print(dml_plr_obj_internal.summary)
               coef   std err         t     P>|t|     2.5 %    97.5 %
        d  0.455418  0.134476  3.386625  0.000708  0.191851  0.718986

    .. code-tab:: r R

        > # R-code here
        > a=5

In the second sample code, we use the K-Folds cross-validator of sklearn :py:class:`~sklearn.model_selection.KFold`
and set the partition via the ``set_sample_splitting()`` method.

.. tabs::

    .. code-tab:: py

        >>> dml_plr_obj_external = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m, draw_sample_splitting = False)

        >>> from sklearn.model_selection import KFold
        >>> np.random.seed(314)
        >>> kf = KFold(n_splits=4, shuffle=True)
        >>> smpls = [[(train, test) for train, test in kf.split(obj_dml_data.x)]]

        >>> dml_plr_obj_external.set_sample_splitting(smpls)
        >>> dml_plr_obj_external.fit()
        >>> print(dml_plr_obj_external.summary)
               coef   std err         t     P>|t|     2.5 %    97.5 %
        d  0.455418  0.134476  3.386625  0.000708  0.191851  0.718986

    .. code-tab:: r R

        > # R-code here
        > a=5

Sample-splitting without cross-fitting
++++++++++++++++++++++++++++++++++++++

The boolean flag ``apply_cross_fitting`` allows to estimate DML models without applying cross-fitting.
It results in randomly splitting the sample into two parts.
The first half of the data is used for the estimation of the nuisance ML models and the second half for estimating the
causal parameter.
Note that cross-fitting performs well empirically and is recommended to remove bias induced by overfitting, see also
:ref:`bias_overfitting`.

.. tabs::

    .. code-tab:: py

        >>> dml_plr_obj_external = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m,
        >>>                                        n_folds = 2, apply_cross_fitting = False)
        >>> dml_plr_obj_external.fit()
        >>> print(dml_plr_obj_external.summary)
               coef  std err         t     P>|t|     2.5 %    97.5 %
        d  0.538687  0.12052  4.469695  0.000008  0.302472  0.774901
        >>> print(dml_plr_obj_external.n_obs)
        100
        >>> print(dml_plr_obj_external.psi.shape)
        (50, 1, 1)

    .. code-tab:: r R

        > # R-code here
        > a=5

Note, that in order to split data unevenly into train and test the interface to externally set the sample splitting
via ``set_sample_splitting()`` needs to be applied, like for example:

.. tabs::

    .. code-tab:: py

        >>> dml_plr_obj_external = dml.DoubleMLPLR(obj_dml_data, ml_g, ml_m,
        >>>                                        n_folds = 2, apply_cross_fitting = False, draw_sample_splitting = False)

        >>> from sklearn.model_selection import train_test_split
        >>> smpls = train_test_split(np.arange(obj_dml_data.n_obs), train_size=0.8)
        >>> smpls = [np.sort(x) for x in smpls]  # only sorted indices are supported
        >>> dml_plr_obj_external.set_sample_splitting([[smpls]])

        >>> dml_plr_obj_external.fit()
        >>> print(dml_plr_obj_external.summary)
              coef  std err         t     P>|t|     2.5 %    97.5 %
        d  0.59366  0.12982  4.572952  0.000005  0.339218  0.848102
        >>> print(dml_plr_obj_external.n_obs)
        100
        >>> print(dml_plr_obj_external.psi.shape)
        (20, 1, 1)

    .. code-tab:: r R

        > # R-code here
        > a=5

