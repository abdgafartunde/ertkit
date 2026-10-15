from __future__ import annotations

import numpy as np
import pytest

from ertkit import DataWarning, ErrorModel


def test_defaults_include_a_modeling_floor() -> None:
    noise = ErrorModel()
    assert noise.relative == 0.03
    assert noise.absolute == 0.0


def test_standard_deviations() -> None:
    noise = ErrorModel(relative=0.02, absolute=1e-3)
    r = np.array([0.01, 0.1, -0.1])
    np.testing.assert_allclose(noise.resistance_std(r), 1e-3 + 0.02 * np.abs(r))
    np.testing.assert_allclose(noise.log_std(r), 0.02 + 1e-3 / np.abs(r))


def test_usable_readings() -> None:
    noise = ErrorModel(relative=0.03, absolute=1e-3)
    r = np.array([1.0, 3e-3, 2.9e-3, 0.0, -1.0, np.nan])
    np.testing.assert_array_equal(
        noise.usable(r), [True, True, False, False, False, False]
    )


@pytest.mark.parametrize(
    ("relative", "absolute", "match"),
    [
        (-0.1, 0.0, "relative"),
        (0.03, np.inf, "absolute"),
        (0.0, 0.0, "at least one"),
    ],
)
def test_validation(relative: float, absolute: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ErrorModel(relative=relative, absolute=absolute)


def test_is_hashable_and_comparable() -> None:
    assert ErrorModel(relative=0.05) == ErrorModel(relative=0.05)
    assert len({ErrorModel(), ErrorModel()}) == 1


def test_a_percentage_given_as_relative_error_is_flagged() -> None:
    with pytest.warns(DataWarning, match="0.03 means 3%"):
        ErrorModel(relative=3.0)
