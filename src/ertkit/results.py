"""Result objects shared by the inversion methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["InversionResult"]


@dataclass(frozen=True, eq=False, kw_only=True, slots=True)
class InversionResult:
    r"""Common part of every inversion result.

    Attributes
    ----------
    predicted : ndarray
        Data predicted by the final model (apparent resistivity, ohm meters).
    chi2 : float
        Final misfit :math:`\chi^2` of the log data.
    chi2_target : float
        The value expected for a model that fits the data to their errors:
        :math:`N_d - p` for a fit with :math:`p` free parameters, :math:`N_d`
        for a regularized inversion.
    n_data : int
        Number of data :math:`N_d`.
    converged : bool
        Whether the method met its convergence criterion.
    stop_reason : str
        Why the iteration stopped.
    n_iterations : int
        Number of iterations (or function evaluations, for methods that count
        those).
    """

    predicted: NDArray[np.float64]
    chi2: float
    chi2_target: float
    n_data: int
    converged: bool
    stop_reason: str
    n_iterations: int

    @property
    def chi2_ratio(self) -> float:
        r""":math:`\chi^2` divided by its target; about 1 for a fit to the noise level."""
        return float(self.chi2 / self.chi2_target)
