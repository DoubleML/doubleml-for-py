from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone, is_classifier, is_regressor
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import _check_sample_weight, check_is_fitted, validate_data


class GlobalRegressor(RegressorMixin, BaseEstimator):
    """
    A global regressor that ignores the attribute `sample_weight` when being fit to ensure a global fit.

    Parameters
    ----------
    base_estimator: regressor implementing ``fit()`` and ``predict()``
        Regressor that is used when ``fit()`` and ``predict()`` are being called.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y, sample_weight=None):
        """
        Fits the regressor provided in ``base_estimator``. Ignores ``sample_weight``.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        Training data.

        y: array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.

        sample_weight: array-like of shape (n_samples,).
        Individual weights for each sample. Ignored.
        """
        if not is_regressor(self.base_estimator):
            raise ValueError(f"base_estimator must be a regressor. Got {self.base_estimator.__class__.__name__} instead.")

        X, y = validate_data(self, X, y)
        _check_sample_weight(sample_weight, X)
        self._fitted_learner = clone(self.base_estimator)
        self._fitted_learner.fit(X, y)

        return self

    def predict(self, X):
        """
        Predicts using the regressor provided in ``base_estimator``.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        Samples.
        """

        check_is_fitted(self)
        return self._fitted_learner.predict(X)


class GlobalClassifier(ClassifierMixin, BaseEstimator):
    """
    A global classifier that ignores the attribute ``sample_weight`` when being fit to ensure a global fit.

    Parameters
    ----------
    base_estimator: classifier implementing ``fit()`` and ``predict_proba()``
        Classifier that is used when ``fit()``, ``predict()`` and ``predict_proba()`` are being called.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y, sample_weight=None):
        """
        Fits the classifier provided in ``base_estimator``. Ignores ``sample_weight``.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        Training data.

        y: array-like of shape (n_samples,) or (n_samples, n_targets)
        Target classes.

        sample_weight: array-like of shape (n_samples,).
        Individual weights for each sample. Ignored.
        """
        if not is_classifier(self.base_estimator):
            raise ValueError(f"base_estimator must be a classifier. Got {self.base_estimator.__class__.__name__} instead.")

        X, y = validate_data(self, X, y)
        _check_sample_weight(sample_weight, X)
        self.classes_ = unique_labels(y)
        self._fitted_learner = clone(self.base_estimator)
        self._fitted_learner.fit(X, y)

        return self

    def predict(self, X):
        """
        Predicts using the classifier provided in ``base_estimator``.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        Samples.
        """
        check_is_fitted(self)
        return self._fitted_learner.predict(X)

    def predict_proba(self, X):
        """
        Probability estimates using the classifier provided in ``base_estimator``.
        The returned estimates for all classes are ordered by the label of classes.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        Samples to be scored.
        """
        check_is_fitted(self)
        return self._fitted_learner.predict_proba(X)
