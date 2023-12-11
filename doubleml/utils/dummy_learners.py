from sklearn.base import BaseEstimator


class dummy_regressor(BaseEstimator):
    """
    A dummy regressor that raises an AttributeError when attempting to access
    its fit, predict, or set_params methods.
    Attributes
    ----------
    _estimator_type : str
        Type of the estimator, set to "regressor".
    Methods
    -------
    fit(*args)
        Raises AttributeError: "Accessed fit method of DummyRegressor!"
    predict(*args)
        Raises AttributeError: "Accessed predict method of DummyRegressor!"
    set_params(*args)
        Raises AttributeError: "Accessed set_params method of DummyRegressor!"
    """

    _estimator_type = "regressor"

    def fit(*args):
        raise AttributeError("Accessed fit method of dummy_regressor!")

    def predict(*args):
        raise AttributeError("Accessed predict method of dummy_regressor!")

    def set_params(*args):
        raise AttributeError("Accessed set_params method of dummy_regressor!")


class dummy_classifier(BaseEstimator):
    """
    A dummy classifier that raises an AttributeError when attempting to access
    its fit, predict, set_params, or predict_proba methods.
    Attributes
    ----------
    _estimator_type : str
        Type of the estimator, set to "classifier".
    Methods
    -------
    fit(*args)
        Raises AttributeError: "Accessed fit method of DummyClassifier!"
    predict(*args)
        Raises AttributeError: "Accessed predict method of DummyClassifier!"
    set_params(*args)
        Raises AttributeError: "Accessed set_params method of DummyClassifier!"
    predict_proba(*args, **kwargs)
        Raises AttributeError: "Accessed predict_proba method of DummyClassifier!"
    """

    _estimator_type = "classifier"

    def fit(*args):
        raise AttributeError("Accessed fit method of dummy_classifier!")

    def predict(*args):
        raise AttributeError("Accessed predict method of dummy_classifier!")

    def set_params(*args):
        raise AttributeError("Accessed set_params method of dummy_classifier!")

    def predict_proba(*args, **kwargs):
        raise AttributeError("Accessed predict_proba method of dummy_classifier!")
