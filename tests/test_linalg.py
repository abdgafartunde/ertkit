from __future__ import annotations

import numpy as np
import pytest

from ertkit._linalg import matmul


def test_matches_the_operator() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((5, 3)), rng.standard_normal((3, 4))
    np.testing.assert_array_equal(matmul(a, b), a @ b)


def test_genuine_overflow_still_warns() -> None:
    a = np.full((2, 2), 1e200)
    with pytest.warns(RuntimeWarning, match="non-finite"):
        matmul(a, a)


def test_non_finite_input_propagates_without_a_new_warning() -> None:
    a = np.array([[np.inf, 1.0], [0.0, 1.0]])
    assert np.isinf(matmul(a, np.eye(2))).any()
