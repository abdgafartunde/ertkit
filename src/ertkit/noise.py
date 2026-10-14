"""Measurement-error models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["ErrorModel"]


@dataclass(frozen=True, kw_only=True, slots=True)
class ErrorModel:
    r"""Standard deviation of measurement errors on the transfer resistance.

    Errors are specified on the transfer resistance :math:`R = \Delta V / I`
    (LaBrecque et al. 1996) as

    .. math::

        s_R = a + b \, |R|,

    with an absolute part :math:`a` in ohms (voltage noise) and a relative part
    :math:`b`. Since apparent resistivity is :math:`\rho_a = K R`, the data
    :math:`d = \ln \rho_a` have log-space standard deviation
    :math:`s = b + a / |R|`.

    Parameters
    ----------
    relative : float, optional
        Relative part :math:`b`. The default, 0.03, is a common choice that
        includes a floor for modeling errors; change it when an error model
        fitted to reciprocal measurements (Tso et al. 2017) justifies another
        value.
    absolute : float, optional
        Absolute part :math:`a`, in ohms. Default 0.

    Raises
    ------
    ValueError
        If a part is negative or not finite, or if both are zero.

    Examples
    --------
    >>> from ertkit import ErrorModel
    >>> noise = ErrorModel(relative=0.03, absolute=1e-4)
    >>> round(float(noise.log_std(0.01)[0]), 6)
    0.04
    """

    relative: float = 0.03
    absolute: float = 0.0

    def __post_init__(self) -> None:
        for name in ("relative", "absolute"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} error must be finite and non-negative")
        if self.relative == 0 and self.absolute == 0:
            raise ValueError("at least one of relative and absolute must be positive")
        object.__setattr__(self, "relative", float(self.relative))
        object.__setattr__(self, "absolute", float(self.absolute))

    def resistance_std(self, resistance: ArrayLike) -> NDArray[np.float64]:
        r"""Standard deviation :math:`s_R = a + b|R|` of transfer resistances.

        Parameters
        ----------
        resistance : array_like
            Transfer resistances :math:`R` in ohms.

        Returns
        -------
        ndarray of float
            Standard deviations in ohms.
        """
        r = np.atleast_1d(np.asarray(resistance, dtype=np.float64))
        return self.absolute + self.relative * np.abs(r)

    def log_std(self, resistance: ArrayLike) -> NDArray[np.float64]:
        r"""Standard deviation :math:`s = b + a/|R|` of :math:`\ln \rho_a`.

        Parameters
        ----------
        resistance : array_like
            Transfer resistances :math:`R` in ohms.

        Returns
        -------
        ndarray of float
            Standard deviations of the natural logarithm of the data.
        """
        r = np.atleast_1d(np.asarray(resistance, dtype=np.float64))
        return np.asarray(self.relative + self.absolute / np.abs(r), dtype=np.float64)

    def usable(self, resistance: ArrayLike) -> NDArray[np.bool_]:
        """Readings that can be inverted in log space.

        A reading is usable when :math:`R > 0` and, if there is an absolute
        part, :math:`|R| \\ge 3a`; below that the logarithm is dominated by
        noise.

        Parameters
        ----------
        resistance : array_like
            Transfer resistances :math:`R` in ohms.

        Returns
        -------
        ndarray of bool
            True for each usable reading.
        """
        r = np.atleast_1d(np.asarray(resistance, dtype=np.float64))
        return np.asarray(np.isfinite(r) & (r > 0) & (np.abs(r) >= 3.0 * self.absolute))
