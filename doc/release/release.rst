:parenttoc: True

Release notes
=============

.. tabbed:: Python

    **DoubleML 0.1.2**

    - Fixed a compatibility issue with `scikit-learn` 0.24, which only affected some unit tests
      (`#70 <https://github.com/DoubleML/doubleml-for-py/issues/70>`_, `#71 <https://github.com/DoubleML/doubleml-for-py/pull/71>`_)
    - Added scheduled unit tests on github-action (three times a week) `#69 <https://github.com/DoubleML/doubleml-for-py/pull/69>`_
    - Split up estimation of nuisance functions and computation of score function components. Further introduced a
      private method `_est_causal_pars_and_se()`, see `#72 <https://github.com/DoubleML/doubleml-for-py/pull/72>`_.
      This is needed for the DoubleML-Serverless project: https://github.com/DoubleML/doubleml-serverless.

    **DoubleML 0.1.1**

    - Bug fix in the drawing of bootstrap weights for the multiple treatment case
      `#66 <https://github.com/DoubleML/doubleml-for-py/pull/66>`_ (see also https://github.com/DoubleML/doubleml-for-r/pull/28)
    - Update install instructions as DoubleML is now listed on pypi
    - Prepare submission to conda-forge: Include LICENSE file in source distribution
    - Documentation is now served with HTTPS `https://docs.doubleml.org/ <https://docs.doubleml.org/>`_

    **DoubleML 0.1.0**

    - Initial release
    - Development at `https://github.com/DoubleML/doubleml-for-py <https://github.com/DoubleML/doubleml-for-py>`_
    - The Python package **DoubleML** provides an implementation of the double / debiased machine learning framework of
      `Chernozhukov et al. (2018) <https://doi.org/10.1111/ectj.12097)>`_.
    - Implements double machine learning for four different models:

        - Partially linear regression models (PLR) in class ``DoubleMLPLR``
        - Partially linear IV regression models (PLIV) in class ``DoubleMLPLIV``
        - Interactive regression models (IRM) in class ``DoubleMLIRM``
        - Interactive IV regression models (IIVM) in class ``DoubleMLIIVM``

    - All model classes are inherited from an abstract base class ``DoubleML`` where the key elements of double machine
      learning are implemented.

.. tabbed:: R

    **DoubleML 0.1.1**

    - First release to CRAN `https://cran.r-project.org/package=DoubleML <https://cran.r-project.org/package=DoubleML>`_
    - Clean up of imports
    - Continuous integration was extended by unit tests on github actions
      `https://github.com/DoubleML/doubleml-for-r/actions <https://github.com/DoubleML/doubleml-for-r/actions>`_

    **DoubleML 0.1.0**

    - Initial release
    - Development at `https://github.com/DoubleML/doubleml-for-r <https://github.com/DoubleML/doubleml-for-r>`_
    - The R package **DoubleML** provides an implementation of the double / debiased machine learning framework of
      `Chernozhukov et al. (2018) <https://doi.org/10.1111/ectj.12097)>`_.
    - Implements double machine learning for four different models:

        - Partially linear regression models (PLR) in class ``DoubleMLPLR``
        - Partially linear IV regression models (PLIV) in class ``DoubleMLPLIV``
        - Interactive regression models (IRM) in class ``DoubleMLIRM``
        - Interactive IV regression models (IIVM) in class ``DoubleMLIIVM``

    - All model classes are inherited from ``DoubleML`` where the key elements of double machine learning are
      implemented.

