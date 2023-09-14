class dummy_regressor:
    _estimator_type = "regressor"

    def fit(*args):
        raise AttributeError("Accessed fit method of dummy_regressor!")

    def predict(*args):
        raise AttributeError("Accessed predict method of dummy_regressor!")

    def set_params(*args):
        print("\n\nAccessed set_params method of dummy_regressor!\n\n")

    def get_params(*args, **kwargs):
        print("\n\nAccessed get_params method of dummy_regressor!\n\n")


class dummy_classifier:
    _estimator_type = "classifier"

    def fit(*args):
        raise AttributeError("Accessed fit method of dummy_classifier!")

    def predict(*args):
        raise AttributeError("Accessed predict method of dummy_classifier!")

    def set_params(*args):
        raise AttributeError("Accessed set_params method of dummy_classifier!")

    def get_params(*args, **kwargs):
        raise AttributeError("Accessed get_params method of dummy_classifier!")
