import numpy as np
from sklearn.tree import DecisionTreeClassifier


def fit_policytree(orth_signal, features, depth):
    policytree_model = DecisionTreeClassifier(max_depth=depth,
                                              ccp_alpha=.01,
                                              min_samples_leaf=8).fit(X=features,
                                                                      y=(np.sign(orth_signal) + 1) / 2,
                                                                      sample_weight=np.abs(orth_signal))

    return policytree_model
