# DoubleML - Double Machine Learning in Python

The Python package **DoubleML** provides an implementation of the double / debiased machine learning framework of
[Chernozhukov et al. (2018)](https://doi.org/10.1111/ectj.12097).
It is built on top of [scikit-learn](https://scikit-learn.org).

Note that the Python package was developed together with an R twin based on [mlr3](https://mlr3.mlr-org.com/).
The R package is available at [https://github.com/DoubleML/doubleml-for-r](https://github.com/DoubleML/doubleml-for-r).

## Installation

### Dependencies

**DoubleML** requires

- Python
- sklearn
- numpy
- scipy
- pandas
- statsmodels
- joblib

We plan to push a first release of the DoubleML package to pip and conda very soon.
Until then we recommend to install from source via

```
git clone git@github.com:DoubleML/doubleml-for-py.git
cd doubleml-for-py
pip install --editable .
```

<!--- TODO: Add a reference to the installation instructions here when the user guide is online. -->
