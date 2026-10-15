"""Checks on the installed package itself: version, typing marker, imports."""

from __future__ import annotations

import re
import subprocess
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

import ertkit


def test_version_matches_metadata() -> None:
    assert ertkit.__version__ == version("ertkit")
    assert ertkit.__version__ != "0+unknown"


def test_py_typed_is_shipped() -> None:
    assert files("ertkit").joinpath("py.typed").is_file()


def test_import_does_not_import_matplotlib() -> None:
    code = "import sys, ertkit; sys.exit('matplotlib' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0


def test_show_versions_lists_packages(capsys: pytest.CaptureFixture[str]) -> None:
    ertkit.show_versions()
    out = capsys.readouterr().out
    for name in ("python", "platform", "ertkit", "numpy", "scipy", "libdlf"):
        assert name in out
    assert ertkit.__version__ in out


def test_missing_package_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from ertkit import _show_versions

    monkeypatch.setattr(_show_versions, "_PACKAGES", ("ertkit", "no-such-package-xyz"))
    assert _show_versions._versions()["no-such-package-xyz"] == "not installed"


def test_readme_quick_start_output_is_current() -> None:
    # The README shows the output of its quick start; run the code and compare,
    # allowing 1% in the numbers for differences between platforms.
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    code = readme.split("## Quick start", 1)[1].split("```python\n", 1)[1]
    code = code.split("```", 1)[0]
    shown = readme.split("The first `print` shows:", 1)[1].split("```text\n", 1)[1]
    shown = shown.split("```", 1)[0]
    printed = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    ).stdout
    printed = printed[: printed.index("depth of investigation")]
    number = r"-?\d+(?:\.\d+)?(?:e[+-]\d+)?"
    assert re.sub(number, "#", printed).strip() == re.sub(number, "#", shown).strip()
    np.testing.assert_allclose(
        [float(x) for x in re.findall(number, printed)],
        [float(x) for x in re.findall(number, shown)],
        rtol=0.01,
    )
