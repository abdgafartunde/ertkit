"""Plots of soundings and layered models.

Needs matplotlib, which the ``plot`` extra installs::

    pip install "ertkit[plot]"
"""

from __future__ import annotations

try:
    import matplotlib  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised in a subprocess test
    raise ImportError(
        'ertkit.plot needs matplotlib; install it with: pip install "ertkit[plot]"'
    ) from exc

from ertkit.plot._ves import layered_model, sounding

__all__ = ["layered_model", "sounding"]
