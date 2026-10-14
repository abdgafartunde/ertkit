"""Warning categories raised by ertkit."""

from __future__ import annotations

__all__ = ["ConvergenceWarning", "DataWarning"]


class ConvergenceWarning(UserWarning):
    """An iterative method stopped before meeting its convergence criterion."""


class DataWarning(UserWarning):
    """Some data were removed or are unreliable (for example near the noise level)."""
