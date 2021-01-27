import numpy as np
from cvxopt.solvers import qp
from cvxopt import matrix

from sklearn.model_selection import KFold

from ._helper import _dml_cv_predict


class DoubleMLEnsemble:
    def __init__(self,
                 learner,
                 n_folds):
        self.n_learner = len(learner)
        self.learner = learner
        self.n_folds = n_folds

    def fit(self, X, y):
        n_obs = len(y)
        smpls = [(train, test) for train, test in KFold().split(np.zeros(n_obs))]
        y_hat = np.zeros((n_obs, self.n_learner))
        for i_learner, this_learner in enumerate(self.learner):
            y_hat[:, i_learner] = _dml_cv_predict(this_learner, X, y, smpls=smpls)

        l = np.zeros(self.n_learner)
        q = np.zeros((self.n_learner, self.n_learner))
        for i_learner in range(self.n_learner):
            l[i_learner] = (-1)*np.mean(np.multiply(y, y_hat[:, i_learner]))
            for j_learner in range(self.n_learner):
                q[i_learner, j_learner] = np.mean(np.multiply(y_hat[:, i_learner], y_hat[:, j_learner]))

        q = matrix(q)
        l = matrix(l)
        I = matrix(np.eye(self.n_learner))
        G = matrix(np.vstack((I, -I)))
        h = matrix(np.hstack((np.ones(self.n_learner), np.zeros(self.n_learner))))
        A = matrix(np.ones((1, self.n_learner)))
        b = matrix(np.ones(1))
        res = qp(q, l, G, h, A, b)

        self.weights = np.array(res['x'])
        for i_learner, this_learner in enumerate(self.learner):
            this_learner.fit(X, y)

        return self

    def predict(self, X):
        n_obs = X.shape[0]
        y_hat = np.zeros((n_obs, self.n_learner))
        for i_learner, this_learner in enumerate(self.learner):
            y_hat[:, i_learner] = this_learner.predict(X)

        preds = np.squeeze(np.matmul(y_hat, self.weights))
        return preds

    def get_params(self, deep=True):
        return dict(learner=self.learner, n_folds=self.n_folds)

    def set_params(self, deep=True):
        return
