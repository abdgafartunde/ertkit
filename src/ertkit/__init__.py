"""ertkit: verified VES and 2.5-D ERT inversion for groundwater exploration."""

from __future__ import annotations

from ertkit._show_versions import show_versions
from ertkit._warnings import ConvergenceWarning, DataWarning
from ertkit.noise import ErrorModel

try:
    from ertkit._version import __version__
except ImportError:  # pragma: no cover - only when the package is not installed
    __version__ = "0+unknown"

__all__ = [
    "ConvergenceWarning",
    "DataWarning",
    "ErrorModel",
    "__version__",
    "show_versions",
]
