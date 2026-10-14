"""Warning categories raised by ertkit."""

from __future__ import annotations

__all__ = ["ConvergenceWarning", "DataWarning"]


class ConvergenceWarning(UserWarning):
    """An iterative method stopped before meeting its convergence criterion."""


class DataWarning(UserWarning):
    """Data were removed, or the data cannot support a result as requested.

    Raised for removed readings, a poor fit, parameters the data do not
    resolve, an undetermined depth of investigation, and errors that appear
    over- or underestimated.
    """
