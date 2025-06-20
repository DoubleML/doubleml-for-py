import numpy as np
import pandas as pd
import pytest

from doubleml.did.utils._did_utils import (
    _check_anticipation_periods,
    _check_control_group,
    _check_gt_combination,
    _check_gt_values,
    _construct_gt_combinations,
    _construct_gt_index,
    _construct_post_treatment_mask,
    _get_id_positions,
    _get_never_treated_value,
    _is_never_treated,
    _set_id_positions,
)


@pytest.mark.ci
def test_get_never_treated_value():
    assert _get_never_treated_value(np.array([1, 2])) == 0
    assert np.isinf(_get_never_treated_value(np.array([1.0, 2.0])))
    assert np.isinf(_get_never_treated_value(np.array([1.0, 2])))
    assert _get_never_treated_value(np.array(["2024-01-01", "2024-01-02"], dtype="datetime64")) is pd.NaT
    assert _get_never_treated_value(np.array(["2024-01-01", "2024-01-02"])) == 0


@pytest.mark.ci
def test_is_never_treated():
    # check single values
    arguments = (
        (0, 0, True),
        (1, 0, False),
        (np.inf, np.inf, True),
        (0, np.inf, False),
        (np.nan, np.inf, False),
        (pd.NaT, pd.NaT, True),
        (0, pd.NaT, False),
    )
    for x, never_treated_value, expected in arguments:
        assert _is_never_treated(x, never_treated_value) == expected

    # check arrays
    arguments = (
        (np.array([0, 1]), 0, np.array([True, False])),
        (np.array([0, 1]), np.inf, np.array([False, False])),
        (np.array([0, 1]), pd.NaT, np.array([False, False])),
        (np.array([0, np.inf]), 0, np.array([True, False])),
        (np.array([0, np.inf]), np.inf, np.array([False, True])),
        (np.array([0, pd.NaT]), 0, np.array([True, False])),
        (np.array([0, pd.NaT]), pd.NaT, np.array([False, True])),
    )
    for x, never_treated_value, expected in arguments:
        assert np.all(_is_never_treated(x, never_treated_value) == expected)


@pytest.mark.ci
def test_check_control_group():
    with pytest.raises(ValueError, match="The control group has to be one of"):
        _check_control_group("invalid_control_group")


@pytest.mark.ci
def test_check_anticipation_periods():
    with pytest.raises(TypeError, match="The anticipation periods must be an integer."):
        _check_anticipation_periods("invalid_type")
    with pytest.raises(ValueError, match="The anticipation periods must be non-negative."):
        _check_anticipation_periods(-1)

    assert _check_anticipation_periods(0) == 0
    assert _check_anticipation_periods(1) == 1


@pytest.mark.ci
def test_check_gt_combination():
    valid_args = {
        "gt_combination": (1, 0, 1),
        "g_values": np.array([-1, 1, 2, np.inf]),
        "t_values": np.array([0, 1, 2]),
        "never_treated_value": np.inf,
        "anticipation_periods": 0,
    }
    invalid_args = [
        (
            {"gt_combination": (3.0, 0, 1)},
            ValueError,
            r"The value 3.0 is not in the set of treatment group values \[-1.  1.  2. inf\].",
        ),
        ({"gt_combination": (1, 0, 3)}, ValueError, r"The value 3 is not in the set of evaluation period values \[0 1 2\]."),
        ({"gt_combination": (1, 3, 1)}, ValueError, r"The value 3 is not in the set of evaluation period values \[0 1 2\]."),
        (
            {"gt_combination": (0, 0, 1), "g_values": np.array([1, 2, 0]), "never_treated_value": 0},
            ValueError,
            r"The never treated group is not allowed as treatment group \(g_value=0\).",
        ),
        (
            {"gt_combination": (1, 1, 1)},
            ValueError,
            "The pre-treatment and evaluation period must be different. Got 1 for both.",
        ),
        (
            {"gt_combination": (-1, 0, 1)},
            ValueError,
            r"The value -1 \(group value\) is not in the set of evaluation period values \[0 1 2\].",
        ),
    ]
    for arg, error, msg in invalid_args:
        with pytest.raises(error, match=msg):
            _check_gt_combination(**(valid_args | arg))

    msg = r"The treatment was assigned before the first pre-treatment period \(including anticipation\)."
    with pytest.warns(UserWarning, match=msg):
        _check_gt_combination(**(valid_args | {"gt_combination": (1, 1, 2)}))
    with pytest.warns(UserWarning, match=msg):
        _check_gt_combination(**(valid_args | {"gt_combination": (1, 0, 1), "anticipation_periods": 1}))


@pytest.mark.ci
def test_input_check_gt_values():
    valid_args = {
        "g_values": np.array([1.0, 2.0]),
        "t_values": np.array([0.0, 1.0, 2.0]),
    }
    invalid_args = [
        ({"g_values": ["test"]}, TypeError, r"Invalid type for g_values: expected one of \(<class 'int'>, <class 'float'>\)."),
        ({"t_values": ["test"]}, TypeError, r"Invalid type for t_values: expected one of \(<class 'int'>, <class 'float'>\)."),
        ({"g_values": np.array([[1.0, 2.0]])}, ValueError, "g_values must be a vector. Number of dimensions is 2."),
        ({"t_values": np.array([[0.0, 1.0, 2.0]])}, ValueError, "t_values must be a vector. Number of dimensions is 2."),
        ({"g_values": None}, TypeError, "Invalid type for g_values."),
        ({"t_values": None}, TypeError, "Invalid type for t_values."),
        ({"t_values": np.array([0.0, 1.0, np.nan])}, ValueError, "t_values contains missing or infinite values."),
        ({"t_values": np.array([0.0, 1.0, np.inf])}, ValueError, "t_values contains missing or infinite values."),
        (
            {"t_values": np.array(["2024-01-01", "2024-01-02", "NaT"], dtype="datetime64")},
            ValueError,
            "t_values contains missing values.",
        ),
        (
            {"g_values": np.array(["test", "test"])},
            ValueError,
            (
                "Invalid data type for g_values: expected one of "
                r"\(<class 'numpy.integer'>, <class 'numpy.floating'>, <class 'numpy.datetime64'>\)."
            ),
        ),
        (
            {"t_values": np.array(["test", "test"])},
            ValueError,
            (
                "Invalid data type for t_values: expected one of "
                r"\(<class 'numpy.integer'>, <class 'numpy.floating'>, <class 'numpy.datetime64'>\)."
            ),
        ),
        (
            {"g_values": np.array(["2024-01-01", "2024-01-02"], dtype="datetime64")},
            ValueError,
            r"g_values and t_values must have the same data type. Got datetime64\[D\] for g_values and float64 for t_values.",
        ),
    ]

    for arg, error, msg in invalid_args:
        with pytest.raises(error, match=msg):
            _check_gt_values(**(valid_args | arg))


@pytest.mark.ci
def test_construct_gt_combinations():
    msg = r"gt_combinations must be one of \['standard', 'all', 'universal'\]. test was passed."
    with pytest.raises(ValueError, match=msg):
        _construct_gt_combinations(
            setting="test",
            g_values=np.array([2, 3]),
            t_values=np.array([1, 2, 3, 4]),
            never_treated_value=np.inf,
            anticipation_periods=0,
        )

    msg = "g_values must be sorted in ascending order."
    with pytest.raises(ValueError, match=msg):
        _construct_gt_combinations(
            setting="standard",
            g_values=np.array([3, 2]),
            t_values=np.array([1, 2, 3, 4]),
            never_treated_value=np.inf,
            anticipation_periods=0,
        )

    msg = "t_values must be sorted in ascending order."
    with pytest.raises(ValueError, match=msg):
        _construct_gt_combinations(
            setting="standard",
            g_values=np.array([1, 2]),
            t_values=np.array([3, 2, 1]),
            never_treated_value=np.inf,
            anticipation_periods=0,
        )

    # too large anticipation periods (no valid combinations)
    msg = (
        "No valid group-time combinations found. "
        r"Please check the treatment group values and time period values \(and anticipation\)."
    )
    with pytest.raises(ValueError, match=msg):
        _construct_gt_combinations(
            setting="standard",
            g_values=np.array([2, 3]),
            t_values=np.array([0, 1, 2, 3]),
            never_treated_value=np.inf,
            anticipation_periods=3,
        )

    # Test standard setting
    standard_combinations = _construct_gt_combinations(
        setting="standard",
        g_values=np.array([2, 3]),
        t_values=np.array([0, 1, 2, 3]),
        never_treated_value=np.inf,
        anticipation_periods=0,
    )
    expected_standard = [
        (2, 0, 1),  # g=2, pre=0 (min of t_previous=0 and t_before_g=0), eval=1
        (2, 1, 2),  # g=2, pre=1 (min of t_previous=1 and t_before_g=1), eval=2
        (2, 1, 3),  # g=2, pre=1 (min of t_previous=2 and t_before_g=1), eval=3
        (3, 0, 1),  # g=3, pre=0 (min of t_previous=0 and t_before_g=0), eval=1
        (3, 1, 2),  # g=3, pre=1 (min of t_previous=1 and t_before_g=1), eval=2
        (3, 2, 3),  # g=3, pre=2 (min of t_previous=2 and t_before_g=2), eval=3
    ]
    assert standard_combinations == expected_standard

    # Test all setting
    all_combinations = _construct_gt_combinations(
        setting="all",
        g_values=np.array([2, 3]),
        t_values=np.array([0, 1, 2, 3]),
        never_treated_value=np.inf,
        anticipation_periods=0,
    )
    expected_all = [
        (2, 0, 1),  # g=2, all pre periods before t_eval=1
        (2, 0, 2),  # g=2, all pre periods before t_eval=2
        (2, 1, 2),
        (2, 0, 3),  # g=2, all pre periods before t_eval=3
        (2, 1, 3),
        (3, 0, 1),  # g=3, all pre periods before t_eval=1
        (3, 0, 2),  # g=3, all pre periods before t_eval=2
        (3, 1, 2),
        (3, 0, 3),  # g=3, all pre periods before t_eval=3
        (3, 1, 3),
        (3, 2, 3),
    ]
    assert all_combinations == expected_all

    # Test universal setting
    universal_combinations = _construct_gt_combinations(
        setting="universal",
        g_values=np.array([2, 3]),
        t_values=np.array([0, 1, 2, 3]),
        never_treated_value=np.inf,
        anticipation_periods=0,
    )
    expected_universal = [
        (2, 1, 0),  # g=2, pre=1, eval=0
        (2, 1, 2),  # g=2, pre=1, eval=2
        (2, 1, 3),  # g=2, pre=1, eval=3
        (3, 2, 0),  # g=3, pre=2, eval=0
        (3, 2, 1),  # g=3, pre=2, eval=1
        (3, 2, 3),  # g=3, pre=2, eval=3
    ]
    assert universal_combinations == expected_universal

    # Test standard setting with anticipation periods
    standard_combinations_anticipation = _construct_gt_combinations(
        setting="standard",
        g_values=np.array([2, 3]),
        t_values=np.array([0, 1, 2, 3]),
        never_treated_value=np.inf,
        anticipation_periods=2,
    )
    expected_standard_anticipation = [
        (3, 0, 3),  # g=3, pre=0 (min of t_previous=0 and t_before_g=0), eval=3 with anticipation 2
    ]
    assert standard_combinations_anticipation == expected_standard_anticipation

    # Test all setting with anticipation periods
    all_combinations_anticipation = _construct_gt_combinations(
        setting="all",
        g_values=np.array([2, 3]),
        t_values=np.array([0, 1, 2, 3]),
        never_treated_value=np.inf,
        anticipation_periods=2,
    )
    expected_all_anticipation = [
        (3, 0, 3),  # g=3, all pre periods before t_eval=3 with anticipation 2
    ]
    assert all_combinations_anticipation == expected_all_anticipation

    # Test universal setting with anticipation periods
    universal_combinations_anticipation = _construct_gt_combinations(
        setting="universal",
        g_values=np.array([2, 3]),
        t_values=np.array([0, 1, 2, 3]),
        never_treated_value=np.inf,
        anticipation_periods=2,
    )
    expected_universal_anticipation = [
        (3, 0, 1),  # g=3, pre=0, eval=1 with anticipation 2
        (3, 0, 2),
        (3, 0, 3),
    ]
    assert universal_combinations_anticipation == expected_universal_anticipation


@pytest.mark.ci
def test_construct_gt_index():
    g_values = np.array([0, 2, 3])
    t_values = np.array([1, 2, 3])
    gt_combinations = [(2, 1, 2), (2, 1, 3), (3, 1, 2)]  # g_val, t_pre, t_eval
    result = _construct_gt_index(gt_combinations, g_values, t_values)
    # Check dimensions
    assert result.shape == (3, 3, 3)

    # Check valid entries
    assert result[1, 0, 1] == 0  # First combination (2, 1, 2)
    assert result[1, 0, 2] == 1  # Second combination (2, 1, 3)
    assert result[2, 0, 1] == 2  # Third combination (3, 1, 2)
    assert result.mask[1, 0, 1] == np.False_
    assert result.mask[1, 0, 2] == np.False_
    assert result.mask[2, 0, 1] == np.False_

    # Check that other entries are masked and contain -1
    assert result.mask[0, 0, 0] == np.True_
    assert result.data[0, 0, 0] == -1

    # Test case 2: Empty combinations
    empty_result = _construct_gt_index([], g_values, t_values)
    assert empty_result.shape == (3, 3, 3)
    assert np.all(empty_result.mask)
    assert np.all(empty_result.data == -1)

    # Test case 3: Single combination
    single_combination = [(2, 1, 2)]
    single_result = _construct_gt_index(single_combination, g_values, t_values)
    assert single_result[1, 0, 1] == 0
    assert np.sum(~single_result.mask) == 1  # Only one unmasked entry

    # Test case 4: Different dimensions
    g_values_large = np.array([0, 1, 2, 3, 4])
    t_values_large = np.array([1, 2, 3, 4])
    large_result = _construct_gt_index(gt_combinations, g_values_large, t_values_large)
    assert large_result.shape == (5, 4, 4)


@pytest.mark.ci
def test_construct_post_treatment_mask():
    # Test case 1: Basic case with integer values
    g_values = np.array([2, 3])
    t_values = np.array([1, 2, 3])
    result = _construct_post_treatment_mask(g_values, t_values)

    # Expected mask pattern for g=2:
    # t_eval=1: False (1 not >= 2)
    # t_eval=2: True (2 not >= 2)
    # t_eval=3: True  (3 >= 2)
    expected_g2 = np.array([[False, True, True]] * len(t_values))
    np.testing.assert_array_equal(result[0], expected_g2)

    # Expected mask pattern for g=3:
    # t_eval=1: False (1 not > 3)
    # t_eval=2: False (2 not > 3)
    # t_eval=3: True (3 >= 3)
    expected_g3 = np.array([[False, False, True]] * len(t_values))
    np.testing.assert_array_equal(result[1], expected_g3)

    # Test case 2: Float values with non-integer treatment times
    g_values = np.array([1.5, 2.5])
    t_values = np.array([1.0, 2.0, 3.0])
    result = _construct_post_treatment_mask(g_values, t_values)

    expected_g1_5 = np.array([[False, True, True]] * len(t_values))
    expected_g2_5 = np.array([[False, False, True]] * len(t_values))
    np.testing.assert_array_equal(result[0], expected_g1_5)
    np.testing.assert_array_equal(result[1], expected_g2_5)

    # Test case 3: Single group
    g_values = np.array([2])
    t_values = np.array([1, 2, 3])
    result = _construct_post_treatment_mask(g_values, t_values)
    assert result.shape == (1, 3, 3)
    np.testing.assert_array_equal(result[0], expected_g2)

    # Test case 4: Single time period
    g_values = np.array([1, 2])
    t_values = np.array([3])
    result = _construct_post_treatment_mask(g_values, t_values)
    assert result.shape == (2, 1, 1)
    np.testing.assert_array_equal(result, np.array([[[True]], [[True]]]))

    # Test case 5: Datetime values
    g_values = np.array(["2020-01-01", "2020-06-01"], dtype="datetime64[D]")
    t_values = np.array(["2020-01-01", "2020-03-01", "2020-12-01"], dtype="datetime64[D]")
    result = _construct_post_treatment_mask(g_values, t_values)

    expected_g1 = np.array([[True, True, True]] * len(t_values))
    expected_g2 = np.array([[False, False, True]] * len(t_values))
    np.testing.assert_array_equal(result[0], expected_g1)
    np.testing.assert_array_equal(result[1], expected_g2)


@pytest.mark.ci
def test_get_id_positions():
    # Test case 1: Normal array with valid positions
    a = np.array([1, 2, 3, 4, 5])
    id_positions = np.array([0, 2, 4])
    expected = np.array([1, 3, 5])
    result = _get_id_positions(a, id_positions)
    np.testing.assert_array_equal(result, expected)

    # Test case 2: 2D array with valid positions
    a_2d = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    id_positions = np.array([1, 3])
    expected_2d = np.array([[3, 4], [7, 8]])
    result_2d = _get_id_positions(a_2d, id_positions)
    np.testing.assert_array_equal(result_2d, expected_2d)

    # Test case 3: None input
    a_none = None
    id_positions = np.array([0, 1, 2])
    result_none = _get_id_positions(a_none, id_positions)
    assert result_none is None


@pytest.mark.ci
def test_set_id_positions():
    # Test case 1: Basic 1D array
    a = np.array([1, 2, 3])
    n_obs = 5
    id_positions = np.array([1, 3, 4])
    fill_value = 0
    expected = np.array([0, 1, 0, 2, 3])
    result = _set_id_positions(a, n_obs, id_positions, fill_value)
    np.testing.assert_array_equal(result, expected)

    # Test case 2: 2D array
    a_2d = np.array([[1, 2], [3, 4], [5, 6]])
    n_obs = 5
    id_positions = np.array([0, 2, 4])
    fill_value = -1
    expected_2d = np.array([[1, 2], [-1, -1], [3, 4], [-1, -1], [5, 6]])
    result_2d = _set_id_positions(a_2d, n_obs, id_positions, fill_value)
    np.testing.assert_array_equal(result_2d, expected_2d)

    # Test case 3: None input
    a_none = None
    n_obs = 3
    id_positions = np.array([0, 1])
    fill_value = 0
    result_none = _set_id_positions(a_none, n_obs, id_positions, fill_value)
    assert result_none is None
