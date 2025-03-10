import pytest

from doubleml.data.utils.panel_data_utils import _is_valid_datetime_unit


@pytest.mark.ci
def test_is_valid_datetime_unit():
    # Test all valid units
    for unit in ["Y", "M", "D", "h", "m", "s", "ms", "us", "ns"]:
        assert _is_valid_datetime_unit(unit) == unit, f"Unit {unit} should be valid and return itself"

    # Test invalid units
    invalid_units = ["", "minutes", "d", "H", "S", "MS", "y", "seconds", "days"]
    for unit in invalid_units:
        with pytest.raises(ValueError, match="Invalid datetime unit."):
            _is_valid_datetime_unit(unit)

    # Test case sensitivity
    assert _is_valid_datetime_unit("m") == "m"  # minute is valid
    assert _is_valid_datetime_unit("M") == "M"  # month is valid

    with pytest.raises(ValueError, match="Invalid datetime unit."):
        _is_valid_datetime_unit("d")  # lowercase day is invalid

    assert _is_valid_datetime_unit("D") == "D"  # uppercase day is valid

    # Test edge cases
    with pytest.raises(ValueError, match="Invalid datetime unit."):
        _is_valid_datetime_unit(None)

    with pytest.raises(ValueError, match="Invalid datetime unit."):
        _is_valid_datetime_unit(123)
