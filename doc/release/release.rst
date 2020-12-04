:parenttoc: True

Release notes
=============

.. tabbed:: Python

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

