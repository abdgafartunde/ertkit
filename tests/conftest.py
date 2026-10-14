"""Shared test configuration: Hypothesis profile and the verification report.

Run ``pytest -m verification --verification-report=PATH`` to write a JSON
file with every verification check: what was measured, the tolerance, and why
that tolerance is appropriate. The documentation renders this file.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from hypothesis import settings
from verification_report import _REPORT, Verification

settings.register_profile("ci", derandomize=True, deadline=None, max_examples=50)
settings.register_profile("dev", deadline=None, max_examples=200)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


@pytest.fixture
def verification(request: pytest.FixtureRequest) -> Verification:
    """Fixture used by tests marked ``verification``."""
    return Verification(request.node.nodeid)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--verification-report",
        metavar="PATH",
        default=None,
        help="write the verification checks to this JSON file",
    )


def pytest_sessionfinish(session: pytest.Session) -> None:
    path = session.config.getoption("--verification-report")
    if not path:
        return
    import ertkit

    data = {
        "ertkit": ertkit.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "checks": [asdict(c) for c in _REPORT.checks],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
