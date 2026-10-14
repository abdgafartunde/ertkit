"""Report the versions of ertkit, Python and the main dependencies."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

_PACKAGES = ("ertkit", "numpy", "scipy", "libdlf", "matplotlib")


def _versions() -> dict[str, str]:
    """Return the versions of Python, the platform and the main packages.

    Returns
    -------
    dict of str to str
        Name to version string; packages that are not installed are reported
        as ``"not installed"``.
    """
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in _PACKAGES:
        try:
            info[name] = version(name)
        except PackageNotFoundError:
            info[name] = "not installed"
    return info


def show_versions() -> None:
    """Print the versions of ertkit, Python and the main dependencies.

    Include this output when you report a problem.

    Examples
    --------
    >>> import ertkit
    >>> ertkit.show_versions()  # doctest: +SKIP
    python      3.13.16
    ...
    """
    info = _versions()
    width = max(len(name) for name in info)
    for name, value in info.items():
        # Printing is the purpose of this function.
        print(f"{name:<{width}}  {value}")  # noqa: T201
