import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier


class DoubleMLPolicyTree:
    """Policy Tree fitting for DoubleML.
    Currently avaivable for IRM models.

    Parameters
    ----------
    orth_signal : :class:`numpy.array`
        The orthogonal signal to be predicted. Has to be of shape ``(n_obs,)``,
        where ``n_obs`` is the number of observations.

    x_vars : :class:`pandas.DataFrame`
        The covariates for estimating the policy tree. Has to have the shape ``(n_obs, d)``,
        where ``n_obs`` is the number of observations and ``d`` is the number of predictors.

    depth : bool
        Indicates whether the basis is constructed for GATEs (dummy-basis).
        Default is ``False``.
    """

    def __init__(self,
                 orth_signal,
                 x_vars,
                 depth):

        if not isinstance(orth_signal, np.ndarray):
            raise TypeError('The signal must be of np.ndarray type. '
                            f'Signal of type {str(type(orth_signal))} was passed.')

        if orth_signal.ndim != 1:
            raise ValueError('The signal must be of one dimensional. '
                             f'Signal of dimensions {str(orth_signal.ndim)} was passed.')

        if not isinstance(x_vars, pd.DataFrame):
            raise TypeError('The basis must be of DataFrame type. '
                            f'Basis of type {str(type(x_vars))} was passed.')

        if not x_vars.columns.is_unique:
            raise ValueError('Invalid pd.DataFrame: '
                             'Contains duplicate column names.')

        self._orth_signal = orth_signal
        self._x_vars = x_vars
        self._depth = depth

        # initialize tree
        self._policy_tree = DecisionTreeClassifier(max_depth = depth)

    def __str__(self):
        class_name = self.__class__.__name__
        header = f'================== {class_name} Object ==================\n'
        fit_summary = str(self.summary)
        res = header + \
            '\n------------------ Fit summary ------------------\n' + fit_summary
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
    def basis(self):
        """
        Basis.
        """
        return self._basis

    @property
    def summary(self):
        """
        A summary for the best linear predictor effect after calling :meth:`fit`.
        """
        col_names = ['coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]']
        if self.blp_model is None:
            df_summary = pd.DataFrame(columns=col_names)
        else:
            summary_stats = {'coef': self.blp_model.params,
                             'std err': self.blp_model.bse,
                             't': self.blp_model.tvalues,
                             'P>|t|': self.blp_model.pvalues,
                             '[0.025': self.blp_model.conf_int()[0],
                             '0.975]': self.blp_model.conf_int()[1]}
            df_summary = pd.DataFrame(summary_stats,
                                      columns=col_names)
        return df_summary

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
        self._blp_model = self._policy_tree.fit(X=self._x_vars, y=bin_signal, 
                                                sample_weight=abs_signal)

        return self

    def plot_tree(self):
        # TODO: Implement plotting for fitted tree
        return None
