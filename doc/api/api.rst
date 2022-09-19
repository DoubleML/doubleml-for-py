Python API reference
====================

Double machine learning data class
----------------------------------

.. currentmodule:: doubleml

.. autosummary::
    :toctree: generated/
    :template: class.rst

    DoubleMLData
    DoubleMLClusterData

Double machine learning models
------------------------------

.. currentmodule:: doubleml

.. autosummary::
    :toctree: generated/
    :template: class.rst

    DoubleMLPLR
    DoubleMLPLIV
    DoubleMLIRM
    DoubleMLIIVM


Datasets module
---------------


Dataset loaders
~~~~~~~~~~~~~~~

.. currentmodule:: doubleml

.. autosummary::
   :toctree: generated/

   datasets.fetch_401K
   datasets.fetch_bonus

Dataset generators
~~~~~~~~~~~~~~~~~~

.. currentmodule:: doubleml

.. autosummary::
   :toctree: generated/

   datasets.make_plr_CCDDHNR2018
   datasets.make_pliv_CHS2015
   datasets.make_irm_data
   datasets.make_iivm_data
   datasets.make_plr_turrell2018
   datasets.make_pliv_multiway_cluster_CKMS2021

Score mixin classes for double machine learning models
------------------------------------------------------

.. currentmodule:: doubleml

.. autosummary::
    :toctree: generated/
    :template: class.rst

    double_ml_score_mixins.LinearScoreMixin
    double_ml_score_mixins.NonLinearScoreMixin
