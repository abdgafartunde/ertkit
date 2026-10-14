"""Recorder for the verification report (see conftest.py)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Check:
    test: str
    name: str
    measured: float
    tolerance: float
    justification: str
    passed: bool


@dataclass
class _Report:
    checks: list[_Check] = field(default_factory=list)


_REPORT = _Report()


class Verification:
    """Records a measured error against its tolerance, then asserts it."""

    def __init__(self, test: str) -> None:
        self._test = test

    def check(
        self, name: str, measured: float, tolerance: float, justification: str
    ) -> None:
        """Record ``measured <= tolerance`` and fail the test if it does not hold.

        Parameters
        ----------
        name : str
            Short description of the quantity, for example "max relative error in V".
        measured : float
            The measured value (an error, normally).
        tolerance : float
            The largest acceptable value.
        justification : str
            Why this tolerance is appropriate.
        """
        measured = float(measured)
        passed = bool(np.isfinite(measured) and measured <= tolerance)
        _REPORT.checks.append(
            _Check(self._test, name, measured, float(tolerance), justification, passed)
        )
        assert passed, f"{name}: measured {measured:.3e} > tolerance {tolerance:.3e}"
