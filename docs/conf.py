"""Sphinx configuration for the ertkit documentation."""

from __future__ import annotations

import sys
from pathlib import Path

import ertkit

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "ertkit"
author = "Abd'gafar Tunde Tiamiyu"
copyright = f"2026, {author}"
release = ertkit.__version__
version = release

extensions = [
    "myst_parser",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "verification_report",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = [
    "_build",
    "_generated",
    "**.ipynb_checkpoints",
    "auto_examples/*.ipynb",
]

# MyST: $...$ and $$...$$ math, task lists in the design document.
myst_enable_extensions = ["dollarmath", "amsmath", "tasklist", "deflist"]
myst_heading_anchors = 3

autosummary_generate = True
templates_path = ["_templates"]
autodoc_typehints = "none"  # numpydoc already gives the types
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}
intersphinx_timeout = 30

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"\.py$",
    "download_all_examples": False,
    "within_subsection_order": "FileNameSortKey",
    "remove_config_comments": True,
}

html_theme = "pydata_sphinx_theme"
html_title = "ertkit documentation"
html_theme_options = {
    "github_url": "https://github.com/abdgafartunde/ertkit",
    "navigation_with_keys": False,
    "show_toc_level": 2,
}

# The weekly link check (not run on every build) retries before failing and
# skips publisher sites that refuse automated requests.
linkcheck_retries = 3
linkcheck_ignore = [
    r"https://doi\.org/.*",
    r"https://(onlinelibrary\.)?wiley\.com/.*",
    r"https://library\.seg\.org/.*",
    r"https://academic\.oup\.com/.*",
]
