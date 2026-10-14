"""ertkit: resistivity inversion for groundwater exploration."""

from __future__ import annotations

import logging

from ertkit._show_versions import show_versions
from ertkit._warnings import ConvergenceWarning, DataWarning
from ertkit.noise import ErrorModel

try:
    from ertkit._version import __version__
except ImportError:  # pragma: no cover - only when the package is not installed
    __version__ = "0+unknown"

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "ConvergenceWarning",
    "DataWarning",
    "ErrorModel",
    "__version__",
    "show_versions",
]
