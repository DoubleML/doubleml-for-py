from sklearn.base import BaseEstimator


class DMLDummyRegressor(BaseEstimator):
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
        raise AttributeError("Accessed fit method of DMLDummyRegressor!")

    def predict(*args):
        raise AttributeError("Accessed predict method of DMLDummyRegressor!")

    def set_params(*args):
        raise AttributeError("Accessed set_params method of DMLDummyRegressor!")


class DMLDummyClassifier(BaseEstimator):
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
        raise AttributeError("Accessed fit method of DMLDummyClassifier!")

    def predict(*args):
        raise AttributeError("Accessed predict method of DMLDummyClassifier!")

    def set_params(*args):
        raise AttributeError("Accessed set_params method of DMLDummyClassifier!")

    def predict_proba(*args, **kwargs):
        raise AttributeError("Accessed predict_proba method of DMLDummyClassifier!")
