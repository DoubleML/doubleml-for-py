# DoubleML - Double Machine Learning in Python

[![build](https://github.com/DoubleML/doubleml-for-py/workflows/build/badge.svg)](https://github.com/DoubleML/doubleml-for-py/actions?query=workflow%3Abuild)
[![PyPI version](https://badge.fury.io/py/DoubleML.svg)](https://badge.fury.io/py/DoubleML)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/doubleml.svg)](https://anaconda.org/conda-forge/doubleml)
[![Python version](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/)

The Python package **DoubleML** provides an implementation of the double / debiased machine learning framework of
[Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097).
It is built on top of [scikit-learn](https://scikit-learn.org) (Pedregosa et al., 2011).

Note that the Python package was developed together with an R twin based on [mlr3](https://mlr3.mlr-org.com/).
The R package is also available on [GitHub](https://github.com/DoubleML/doubleml-for-r) and 
[![CRAN Version](https://www.r-pkg.org/badges/version/DoubleML)](https://cran.r-project.org/package=DoubleML).

## Documentation and maintenance

Documentation and website: [https://docs.doubleml.org/](https://docs.doubleml.org/)

**DoubleML** is currently maintained by
[@MalteKurz](https://github.com/MalteKurz) and
[@PhilippBach](https://github.com/PhilippBach).

Bugs can be reported to the issue tracker at
[https://github.com/DoubleML/doubleml-for-py/issues](https://github.com/DoubleML/doubleml-for-py/issues).

## Main Features

Double / debiased machine learning [(Chernozhukov et al. (2018))](https://doi.org/10.1111/ectj.12097) for 

- Partially linear regression models (PLR)
- Partially linear IV regression models (PLIV)
- Interactive regression models (IRM)
- Interactive IV regression models (IIVM)

The object-oriented implementation of DoubleML is very flexible.
The model classes `DoubleMLPLR`, `DoubleMLPLIV`, `DoubleMLIRM` and `DoubleIIVM` implement the estimation of the nuisance
functions via machine learning methods and the computation of the Neyman orthogonal score function.
All other functionalities are implemented in the abstract base class `DoubleML`.
In particular functionalities to estimate double machine learning models and to perform statistical inference via the
methods `fit`, `bootstrap`, `confint`, `p_adjust` and `tune`.
This object-oriented implementation allows a high flexibility for the model specification in terms of ...

- ... the machine learners for the nuisance functions,
- ... the resampling schemes,
- ... the double machine learning algorithm,
- ... the Neyman orthogonal score functions,
- ... 

It further can be readily extended with regards to

- ... new model classes that come with Neyman orthogonal score functions being linear in the target parameter,
- ... alternative score functions via callables,
- ... alternative resampling schemes,
- ... 

![An overview of the OOP structure of the DoubleML package is given in the graphic available at https://github.com/DoubleML/doubleml-for-py/blob/master/doc/oop.svg](/doc/oop.svg?raw=true)

## Installation

**DoubleML** requires

- Python
- sklearn
- numpy
- scipy
- pandas
- statsmodels
- joblib

To install DoubleML with pip use

```
pip install -U DoubleML
```

DoubleML can be installed from source via

```
git clone git@github.com:DoubleML/doubleml-for-py.git
cd doubleml-for-py
pip install --editable .
```

Detailed [installation instructions](https://docs.doubleml.org/stable/intro/install.html) can be found in the documentation.

## Citation

If you use the DoubleML package a citation is highly appreciated:

Bach, P., Chernozhukov, V., Kurz, M. S., and Spindler, M. (2020),
DoubleML - Double Machine Learning in Python.
URL: [https://github.com/DoubleML/doubleml-for-py](https://github.com/DoubleML/doubleml-for-py),
Python-Package version 0.1.2.

Bibtex-entry:

```
@Manual{DoubleML2020,
  title = {DoubleML - Double Machine Learning in Python},
  author = {Bach, P., Chernozhukov, V., Kurz, M. S., and Spindler, M.},
  year = {2020},
  note = {URL: \url{https://github.com/DoubleML/doubleml-for-py}, Python-Package version 0.1.2}
}
```


## References

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. and Robins, J. (2018),
Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21: C1-C68. doi:[10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097).

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, E. (2011),
Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12: 2825--2830, [https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html).
