"""Small linear-algebra helpers."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray


def matmul(a: NDArray[Any], b: NDArray[Any]) -> NDArray[Any]:
    """Matrix product that checks its result instead of the floating-point flags.

    With numpy 2.2 and earlier on recent macOS, the Accelerate BLAS can set
    floating-point flags (divide by zero, overflow) during a matrix product of
    ordinary finite values, which numpy then reports as spurious warnings. The
    flags are therefore ignored during the product, and a genuine problem is
    detected from the result: if finite inputs give a non-finite result, a
    ``RuntimeWarning`` is raised.
    """
    with np.errstate(all="ignore"):
        out = np.asarray(a @ b)
    if (
        not np.all(np.isfinite(out))
        and np.all(np.isfinite(a))
        and np.all(np.isfinite(b))
    ):
        warnings.warn(
            "non-finite values in a matrix product", RuntimeWarning, stacklevel=2
        )
    return out
