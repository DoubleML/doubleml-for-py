"""
===============
DML: Bonus Data
===============
This example shows
TODO: Add a general description!
"""
print(__doc__)

# %%

import numpy as np
import doubleml as dml
from doubleml.datasets import fetch_bonus

from sklearn.linear_model import Lasso, LogisticRegression
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
# Specify learner and estimate causal parameter: PLR model with random forest as learner
# --------------------------------------------------------------------------------------

# Set machine learning methods for m & g
ml_g = RandomForestRegressor()
ml_m = RandomForestRegressor()
n_folds = 2
n_rep = 10

np.random.seed(3141)
dml_plr_rf = dml.DoubleMLPLR(dml_data,
                             ml_g,
                             ml_m,
                             n_folds,
                             n_rep,
                             'partialling out',
                             'dml2')

# set some hyperparameters for the learners
pars = {'n_estimators': 500,
        'max_features': 'sqrt',
        'max_depth': 5}

dml_plr_rf.set_ml_nuisance_params('ml_g', 'tg', pars)
dml_plr_rf.set_ml_nuisance_params('ml_m', 'tg', pars)


# %%
#

dml_plr_rf.fit()
dml_plr_rf.summary

# %%
#

# Load data with polynomial features
dml_data_lasso = dml.datasets.fetch_bonus(polynomial_features=True)
print(dml_data_lasso)

# %%
# Specify learner and estimate causal parameter: PLR model with Lasso as learner
# ------------------------------------------------------------------------------

# Set machine learning methods for m & g
ml_g = Lasso()
ml_m = Lasso()
n_folds = 2
n_rep = 10

np.random.seed(3141)
dml_plr_lasso = dml.DoubleMLPLR(dml_data_lasso,
                                ml_g,
                                ml_m,
                                n_folds,
                                n_rep,
                                'partialling out',
                                'dml2')

# set some hyperparameters for the learners
dml_plr_lasso.set_ml_nuisance_params('ml_g', 'tg', {'alpha': 0.0005})
dml_plr_lasso.set_ml_nuisance_params('ml_m', 'tg', {'alpha': 0.0026})

# %%
#

dml_plr_lasso.fit()
dml_plr_lasso.summary

# %%
# Specify learner and estimate causal parameter: IRM model with random forest as learner
# --------------------------------------------------------------------------------------

# Set machine learning methods for m & g
ml_g = RandomForestRegressor()
ml_m = RandomForestClassifier()
n_folds = 2
n_rep = 10

np.random.seed(3141)
dml_irm_rf = dml.DoubleMLIRM(dml_data,
                             ml_g,
                             ml_m,
                             n_folds,
                             n_rep,
                             'ATE',
                             'dml2')

# set some hyperparameters for the learners
pars = {'n_estimators': 500,
        'max_features': 'sqrt',
        'max_depth': 5}

dml_irm_rf.set_ml_nuisance_params('ml_g0', 'tg', pars)
dml_irm_rf.set_ml_nuisance_params('ml_g1', 'tg', pars)
dml_irm_rf.set_ml_nuisance_params('ml_m', 'tg', pars)

# %%
#

dml_irm_rf.fit()
dml_irm_rf.summary



# %%
# Specify learner and estimate causal parameter: IRM model with Lasso as learner
# ------------------------------------------------------------------------------

# Set machine learning methods for m & g
ml_g = Lasso()
ml_m = LogisticRegression()
np.random.seed(1234)
n_folds = 2
n_rep = 10

np.random.seed(3141)
dml_irm_lasso = dml.DoubleMLIRM(dml_data_lasso,
                                ml_g,
                                ml_m,
                                n_folds,
                                n_rep,
                                'ATE',
                                'dml2')

# set some hyperparameters for the learners
dml_irm_lasso.set_ml_nuisance_params('ml_g0', 'tg', {'alpha': 0.0019})
dml_irm_lasso.set_ml_nuisance_params('ml_g1', 'tg', {'alpha': 0.0073})
dml_irm_lasso.set_ml_nuisance_params('ml_m', 'tg', {'C': 0.0001})

# %%
#

dml_irm_lasso.fit()
dml_irm_lasso.summary

