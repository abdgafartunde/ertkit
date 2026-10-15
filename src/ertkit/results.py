"""Result objects shared by the inversion methods."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import NDArray

__all__ = ["InversionResult"]


@dataclass(frozen=True, eq=False, kw_only=True, slots=True)
class InversionResult:
    r"""Common part of every inversion result.

    Results are immutable: their arrays are read-only.

    Attributes
    ----------
    predicted : ndarray
        Data predicted by the final model (apparent resistivity, ohm meters).
    residuals : ndarray
        Normalized residuals :math:`(d_i - f_i)/s_i` of the log data, one per
        reading; their squares sum to ``chi2``. For a fit to the noise level
        they scatter around 0 with a spread of about 1, without trends.
    chi2 : float
        Final misfit :math:`\chi^2` of the log data.
    chi2_target : float
        The misfit the method aimed at: :math:`N_d - p`, the value expected
        for a fit with :math:`p` free parameters that matches the data to
        their errors; :math:`N_d` for a smooth inversion, or 1.1 times the
        smallest misfit when :math:`N_d` cannot be reached.
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
    residuals: NDArray[np.float64]
    chi2: float
    chi2_target: float
    n_data: int
    converged: bool
    stop_reason: str
    n_iterations: int

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    @property
    def chi2_ratio(self) -> float:
        r""":math:`\chi^2` divided by its target; about 1 for a fit to the noise level."""
        return float(self.chi2 / self.chi2_target)

    def summary(self) -> str:
        """Readable summary of the fit and the model.

        Uncertainty factors are written ``x/ f``: the value multiplied or
        divided by ``f``.
        """
        lines = [
            (
                f"{type(self).__name__}: chi2 = {self.chi2:.4g} for {self.n_data} data "
                f"(target {self.chi2_target:.4g}, ratio {self.chi2_ratio:.3g})"
            ),
            f"converged: {self.converged} ({self.stop_reason})",
        ]
        return "\n".join(lines + self._details())

    def _details(self) -> list[str]:
        """Method-specific lines of the summary."""
        return []
