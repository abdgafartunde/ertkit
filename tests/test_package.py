"""Checks on the installed package itself: version, typing marker, imports."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import version
from importlib.resources import files

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
