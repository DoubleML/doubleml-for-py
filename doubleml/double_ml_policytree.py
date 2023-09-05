import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils.validation import check_is_fitted


class DoubleMLPolicyTree:
    """Policy Tree fitting for DoubleML.
    Currently avaivable for IRM models.

    Parameters
    ----------
    orth_signal : :class:`numpy.array`
        The orthogonal signal to be predicted. Has to be of shape ``(n_obs,)``,
        where ``n_obs`` is the number of observations.

    features : :class:`pandas.DataFrame`
        The covariates for estimating the policy tree. Has to have the shape ``(n_obs, d)``,
        where ``n_obs`` is the number of observations and ``d`` is the number of predictors.

    depth : int
        The depth of the policy tree that will be built. Default is ``2``.

    **tree_params : dict
        Parameters that are forwarded to the :class:`sklearn.tree.DecisionTreeClassifier`.
        Note that by default we perform minimal pruning by setting the ``ccp_alpha = 0.01`` and
        ``min_samples_leaf = 8``. This can be adjusted.

    """

    def __init__(self,
                 orth_signal,
                 features,
                 depth=2,
                 **tree_params):

        if not isinstance(orth_signal, np.ndarray):
            raise TypeError('The signal must be of np.ndarray type. '
                            f'Signal of type {str(type(orth_signal))} was passed.')

        if orth_signal.ndim != 1:
            raise ValueError('The signal must be of one dimensional. '
                             f'Signal of dimensions {str(orth_signal.ndim)} was passed.')

        if not isinstance(features, pd.DataFrame):
            raise TypeError('The features must be of DataFrame type. '
                            f'Features of type {str(type(features))} was passed.')

        if not features.columns.is_unique:
            raise ValueError('Invalid pd.DataFrame: '
                             'Contains duplicate column names.')

        self._orth_signal = orth_signal
        self._features = features
        self._depth = depth
        self._tree_params = tree_params

        self._tree_params.setdefault("ccp_alpha", .01)
        self._tree_params.setdefault("min_samples_leaf", 8)

        # initialize tree
        self._policy_tree = DecisionTreeClassifier(max_depth=self._depth,
                                                   **self._tree_params)

    def __str__(self):
        class_name = self.__class__.__name__
        header = f'================== {class_name} Object ==================\n'
        fit_summary = str(self.summary)
        res = header + \
            '\n------------------ Summary ------------------\n' + fit_summary
        return res

    @property
    def policy_tree(self):
        """
        Policy tree model.
        """
        return self._policy_tree

    @property
    def orth_signal(self):
        """
        Orthogonal signal.
        """
        return self._orth_signal

    @property
    def features(self):
        """
        Covariates.
        """
        return self._features

    @property
    def summary(self):
        """
        A summary for the policy tree.
        """
        summary = pd.DataFrame({"Decision Variables": self._features.keys(), "Max Depth": self._depth})
        return summary

    def fit(self):
        """
        Estimate DoubleMLPolicyTree models.

        Returns
        -------
        self : object
        """
        bin_signal = (np.sign(self._orth_signal) + 1) / 2
        abs_signal = np.abs(self._orth_signal)

        # fit the tree with target binary score, sample weights absolute score and
        # provided feature variables
        self._policy_tree.fit(X=self._features, y=bin_signal,
                              sample_weight=abs_signal)

        return self

    def plot_tree(self):
        """
        Plots the DoubleMLPolicyTree.

        Returns
        -------
        self : object
        """
        check_is_fitted(self._policy_tree, msg='Policy Tree not yet fitted. Call fit before plot_tree.')

        artists = plot_tree(self.policy_tree, feature_names=list(self._features.keys()), filled=True,
                            class_names=["No Treatment", "Treatment"], impurity=False)
        return artists

    def predict(self, features):
        """
        Predicts policy based on the DoubleMLPolicyTree.

        Parameters
        ----------
        features : :class:`pandas.DataFrame`
            The covariates for predicting based on the policy tree. Has to have the shape ``(n_obs, d)``,
            where ``n_obs`` is the number of observations and ``d`` is the number of predictors. Has to
            have the identical keys as the original covariates.

        Returns
        -------
        self : object
        """
        check_is_fitted(self._policy_tree, msg='Policy Tree not yet fitted. Call fit before predict.')

        if not isinstance(features, pd.DataFrame):
            raise TypeError('The features must be of DataFrame type. '
                            f'Features of type {str(type(features))} was passed.')

        if not set(features.keys()) == set(self._features.keys()):
            raise KeyError(f'The features must have the keys {self._features.keys()}. '
                           f'Features with keys {features.keys()} were passed.')

        predictions = self.policy_tree.predict(features)

        return features.assign(pred_treatment=predictions.astype(int))
