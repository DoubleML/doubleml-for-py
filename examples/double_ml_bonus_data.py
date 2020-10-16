"""
===============
DML: Bonus Data
===============
This example shows
TODO: Add a general description!
"""
print(__doc__)

# %%

import doubleml as dml
from doubleml.datasets import fetch_bonus

from sklearn.base import clone
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns

# %%
#

plt.rcParams['figure.figsize'] = 14, 6
sns.set()

# %%
# Load bonus data using the dml datasets module
# ---------------------------------------------

dml_data = dml.datasets.fetch_bonus()
dml_data.data.head()


# %%
# Specify learner and estimate causal parameter
# ---------------------------------------------

# Set machine learning methods for m & g
learner = RandomForestRegressor(max_depth=2, n_estimators=100)
ml_g = clone(learner)
ml_m = clone(learner)
n_folds = 2
n_rep = 100

dml_plr_obj_rf = dml.DoubleMLPLR(dml_data,
                                 ml_g,
                                 ml_m,
                                 n_folds,
                                 n_rep,
                                 'IV-type',
                                 'dml1')

# %%
#

dml_plr_obj_rf.fit()
dml_plr_obj_rf.summary

# %%
#

# Load data with polynomial features
dml_data_lasso = dml.datasets.fetch_bonus(polynomial_features=True)
dml_data_lasso

# %%
#

# Set machine learning methods for m & g
learner = Lasso(alpha=0.1)
ml_g = clone(learner)
ml_m = clone(learner)
n_folds = 2
n_rep = 100

dml_plr_obj_lasso = dml.DoubleMLPLR(dml_data_lasso,
                                    ml_g,
                                    ml_m,
                                    n_folds,
                                    n_rep,
                                    'partialling out',
                                    'dml2')

# %%
#

dml_plr_obj_lasso.fit()
dml_plr_obj_lasso.summary

# %%
#

# Set machine learning methods for m & g
ml_g = RandomForestRegressor(max_depth=2, n_estimators=100)
ml_m = RandomForestClassifier(max_depth=2, n_estimators=100)
n_folds = 2
n_rep=100

dml_irm_obj = dml.DoubleMLIRM(dml_data,
                              ml_g,
                              ml_m,
                              n_folds,
                              n_rep,
                              'ATE',
                              'dml2')

# %%
#

dml_irm_obj.fit()
dml_irm_obj.summary
