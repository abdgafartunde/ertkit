from __future__ import annotations

import numpy as np
import pytest

from ertkit.arrays import geometric_factor


def test_wenner() -> None:
    a = np.array([1.0, 2.5, 10.0])
    np.testing.assert_allclose(
        geometric_factor(a, 2 * a, 2 * a, a), 2 * np.pi * a, rtol=1e-14
    )


def test_schlumberger_finite_mn() -> None:
    s, b = 10.0, 1.0  # AB/2 and MN/2
    k = geometric_factor(s - b, s + b, s + b, s - b)
    np.testing.assert_allclose(k, np.pi * (s**2 - b**2) / (2 * b), rtol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3, 6])
def test_dipole_dipole(n: int) -> None:
    a = 2.0  # A at 0, B at a, M at (n+1)a, N at (n+2)a
    k = geometric_factor((n + 1) * a, n * a, (n + 2) * a, (n + 1) * a)
    np.testing.assert_allclose(abs(k), np.pi * n * (n + 1) * (n + 2) * a, rtol=1e-12)
    assert k < 0  # V_M < V_N: M is nearer than N to the current sink B


def test_pole_dipole_with_remote_electrode() -> None:
    a, n = 1.0, 2  # A at 0, B remote, M at n a, N at (n+1) a
    k = geometric_factor(n * a, np.inf, (n + 1) * a, np.inf)
    np.testing.assert_allclose(k, 2 * np.pi * n * (n + 1) * a, rtol=1e-14)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_rejects_non_positive_distances(bad: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        geometric_factor(bad, 2.0, 2.0, 1.0)


def test_rejects_configuration_without_signal() -> None:
    # M and N equidistant from both current electrodes: no potential difference.
    with pytest.raises(ValueError, match="no potential difference"):
        geometric_factor(1.0, 1.0, 1.0, 1.0)
