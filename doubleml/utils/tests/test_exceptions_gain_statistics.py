import pytest
import numpy as np

from doubleml.utils.gain_statistics import gain_statistics


class test_framework():
    def __init__(self, sensitivity_elements):
        self.sensitivity_elements = sensitivity_elements


class test_dml_class():
    def __init__(self, sensitivity_elements, all_coef):
        self.framework = test_framework(sensitivity_elements)
        self.all_coef = all_coef


n_obs = 1
n_rep = 3
n_coef = 5


@pytest.mark.ci
def test_doubleml_exception_data():
    dml_correct = test_dml_class(
        sensitivity_elements={
            'sigma2': np.random.normal(size=(n_obs, n_rep, n_coef)),
            'nu2': np.random.normal(size=(n_obs, n_rep, n_coef))
        },
        all_coef=np.random.normal(size=(n_rep, n_coef))
    )

    # incorrect types
    dml_incorrect = test_dml_class(
            sensitivity_elements=np.random.normal(size=(n_obs, n_rep, n_coef)),
            all_coef=np.random.normal(size=(n_rep, n_coef))
        )
    msg = r"dml_long does not contain the necessary sensitivity elements\. "
    msg += r"Expected dict for dml_long\.framework\.sensitivity_elements\."
    with pytest.raises(TypeError, match=msg):
        _ = gain_statistics(dml_incorrect, dml_correct)
    msg = r"dml_short does not contain the necessary sensitivity elements\. "
    msg += r"Expected dict for dml_short\.framework\.sensitivity_elements\."
    with pytest.raises(TypeError, match=msg):
        _ = gain_statistics(dml_correct, dml_incorrect)

    # incorrect keys
    dml_incorrect = test_dml_class(
            sensitivity_elements={
                'sigma2': np.random.normal(size=(n_obs, n_rep, n_coef)),
            },
            all_coef=np.random.normal(size=(n_rep, n_coef))
        )
    msg = r"dml_long does not contain the necessary sensitivity elements\. Required keys are: \['sigma2', 'nu2'\]"
    with pytest.raises(ValueError, match=msg):
        _ = gain_statistics(dml_incorrect, dml_correct)
    msg = r"dml_short does not contain the necessary sensitivity elements\. Required keys are: \['sigma2', 'nu2'\]"
    with pytest.raises(ValueError, match=msg):
        _ = gain_statistics(dml_correct, dml_incorrect)

    # incorrect type for keys
    dml_incorrect = test_dml_class(
            sensitivity_elements={
                'sigma2': {},
                'nu2': np.random.normal(size=(n_obs, n_rep, n_coef))
            },
            all_coef=np.random.normal(size=(n_rep, n_coef))
        )
    msg = r"dml_long does not contain the necessary sensitivity elements\. Expected numpy\.ndarray for key sigma2\."
    with pytest.raises(TypeError, match=msg):
        _ = gain_statistics(dml_incorrect, dml_correct)
    msg = r"dml_short does not contain the necessary sensitivity elements\. Expected numpy\.ndarray for key sigma2\."
    with pytest.raises(TypeError, match=msg):
        _ = gain_statistics(dml_correct, dml_incorrect)

    dml_incorrect = test_dml_class(
        sensitivity_elements={
            'sigma2': np.random.normal(size=(n_obs, n_rep, n_coef)),
            'nu2': {}
        },
        all_coef=np.random.normal(size=(n_rep, n_coef))
    )
    msg = r"dml_long does not contain the necessary sensitivity elements\. Expected numpy\.ndarray for key nu2\."
    with pytest.raises(TypeError, match=msg):
        _ = gain_statistics(dml_incorrect, dml_correct)
    msg = r"dml_short does not contain the necessary sensitivity elements\. Expected numpy\.ndarray for key nu2\."
    with pytest.raises(TypeError, match=msg):
        _ = gain_statistics(dml_correct, dml_incorrect)

    # incorrect shape for keys
    dml_incorrect = test_dml_class(
            sensitivity_elements={
                'sigma2': np.random.normal(size=(n_obs + 1, n_rep, n_coef)),
                'nu2': np.random.normal(size=(n_obs, n_rep, n_coef))
            },
            all_coef=np.random.normal(size=(n_rep, n_coef))
        )
    msg = (r"dml_long does not contain the necessary sensitivity elements\. "
           r"Expected 3 dimensions of shape \(1, n_coef, n_rep\) for key sigma2\.")
    with pytest.raises(ValueError, match=msg):
        _ = gain_statistics(dml_incorrect, dml_correct)
    msg = (r"dml_short does not contain the necessary sensitivity elements\. "
           r"Expected 3 dimensions of shape \(1, n_coef, n_rep\) for key sigma2\.")
    with pytest.raises(ValueError, match=msg):
        _ = gain_statistics(dml_correct, dml_incorrect)

    dml_incorrect = test_dml_class(
            sensitivity_elements={
                'sigma2': np.random.normal(size=(n_obs, n_rep, n_coef)),
                'nu2': np.random.normal(size=(n_obs + 1, n_rep, n_coef))
            },
            all_coef=np.random.normal(size=(n_rep, n_coef))
        )
    msg = (r"dml_long does not contain the necessary sensitivity elements\. "
           r"Expected 3 dimensions of shape \(1, n_coef, n_rep\) for key nu2\.")
    with pytest.raises(ValueError, match=msg):
        _ = gain_statistics(dml_incorrect, dml_correct)
    msg = (r"dml_short does not contain the necessary sensitivity elements\. "
           r"Expected 3 dimensions of shape \(1, n_coef, n_rep\) for key nu2\.")
    with pytest.raises(ValueError, match=msg):
        _ = gain_statistics(dml_correct, dml_incorrect)

    # conflicting shape for keys
    dml_incorrect = test_dml_class(
            sensitivity_elements={
                'sigma2': np.random.normal(size=(n_obs, n_rep + 1, n_coef)),
                'nu2': np.random.normal(size=(n_obs, n_rep, n_coef))
            },
            all_coef=np.random.normal(size=(n_rep, n_coef))
        )
    msg = r"dml_long and dml_short do not contain the same shape of sensitivity elements\. "
    msg += r"Shapes of sigma2 are: \(1, 4, 5\) and \(1, 3, 5\)"
    with pytest.raises(ValueError, match=msg):
        _ = gain_statistics(dml_incorrect, dml_correct)
    msg = r"dml_long and dml_short do not contain the same shape of sensitivity elements\. "
    msg += r"Shapes of sigma2 are: \(1, 3, 5\) and \(1, 4, 5\)"
    with pytest.raises(ValueError, match=msg):
        _ = gain_statistics(dml_correct, dml_incorrect)

    dml_incorrect = test_dml_class(
            sensitivity_elements={
                'sigma2': np.random.normal(size=(n_obs, n_rep, n_coef)),
                'nu2': np.random.normal(size=(n_obs, n_rep + 1, n_coef))
            },
            all_coef=np.random.normal(size=(n_rep, n_coef))
        )
    msg = r"dml_long and dml_short do not contain the same shape of sensitivity elements\. "
    msg += r"Shapes of nu2 are: \(1, 4, 5\) and \(1, 3, 5\)"
    with pytest.raises(ValueError, match=msg):
        _ = gain_statistics(dml_incorrect, dml_correct)
    msg = r"dml_long and dml_short do not contain the same shape of sensitivity elements\. "
    msg += r"Shapes of nu2 are: \(1, 3, 5\) and \(1, 4, 5\)"
    with pytest.raises(ValueError, match=msg):
        _ = gain_statistics(dml_correct, dml_incorrect)

    # incorrect type for all_coef
    dml_incorrect = test_dml_class(
            sensitivity_elements={
                'sigma2': np.random.normal(size=(n_obs, n_rep, n_coef)),
                'nu2': np.random.normal(size=(n_obs, n_rep, n_coef))
            },
            all_coef={}
        )
    msg = r"dml_long\.all_coef does not contain the necessary coefficients\. Expected numpy\.ndarray\."
    with pytest.raises(TypeError, match=msg):
        _ = gain_statistics(dml_incorrect, dml_correct)
    msg = r"dml_short\.all_coef does not contain the necessary coefficients\. Expected numpy\.ndarray\."
    with pytest.raises(TypeError, match=msg):
        _ = gain_statistics(dml_correct, dml_incorrect)

    # incorrect shape for all_coef
    dml_incorrect = test_dml_class(
            sensitivity_elements={
                'sigma2': np.random.normal(size=(n_obs, n_rep, n_coef)),
                'nu2': np.random.normal(size=(n_obs, n_rep, n_coef))
            },
            all_coef=np.random.normal(size=(n_rep, n_coef + 1))
        )
    msg = r"dml_long\.all_coef does not contain the necessary coefficients\. Expected shape: \(3, 5\)"
    with pytest.raises(ValueError, match=msg):
        _ = gain_statistics(dml_incorrect, dml_correct)
    msg = r"dml_short\.all_coef does not contain the necessary coefficients\. Expected shape: \(3, 5\)"
    with pytest.raises(ValueError, match=msg):
        _ = gain_statistics(dml_correct, dml_incorrect)
