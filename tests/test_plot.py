"""Smoke tests for the optional plotting module (needs matplotlib)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from ertkit import ErrorModel
from ertkit.ves import LayeredModel, Sounding, Survey, invert_few_layer, invert_smooth

pytest.importorskip("matplotlib")

# matplotlib 3.10 (the oldest supported) calls pyparsing functions that newer
# pyparsing versions deprecate (reported from matplotlib); those are not ertkit's.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning:matplotlib")

import matplotlib as mpl

mpl.use("Agg")

from ertkit import plot

MODEL = LayeredModel(resistivity=[150.0, 30.0, 800.0], thickness=[2.0, 15.0])


@pytest.fixture(scope="module")
def data() -> Sounding:
    ab2 = np.geomspace(1.0, 200.0, 20)
    survey = Survey.schlumberger(ab2=ab2, mn2=np.where(ab2 < 15, 0.5, 5.0))
    return Sounding.synthetic(MODEL, survey, ErrorModel(relative=0.03), rng=1)


def test_sounding_plot(data: Sounding) -> None:
    fit = invert_few_layer(data, 3, rng=0)
    ax = plot.sounding(data, fit, labels=["three layers"])
    # Data, plus one line per MN segment for the fit.
    assert len(ax.get_lines()) >= 2
    assert ax.get_xscale() == "log"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "three layers",
        "data",
    ]


def test_layered_model_plot(data: Sounding) -> None:
    fit = invert_few_layer(data, 3, rng=0)
    smooth = invert_smooth(data)
    ax = plot.layered_model(MODEL, fit, smooth, labels=["true", "few-layer", "smooth"])
    assert ax.get_ylim()[0] > ax.get_ylim()[1]  # depth increases downwards
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "below depth of investigation (smooth)" in texts
    # The default depth range follows the smooth fit's depth of investigation.
    assert ax.get_ylim()[0] == pytest.approx(1.5 * smooth.doi)


def test_log_depth_axis(data: Sounding) -> None:
    smooth = invert_smooth(data, depth_of_investigation=False)
    fit = invert_few_layer(data, 3, rng=0)
    ax = plot.layered_model(smooth, fit, log_depth=True, depth_max=100.0)
    assert ax.get_yscale() == "log"
    assert 0.0 < ax.get_ylim()[1] < ax.get_ylim()[0] == 100.0


def test_missing_matplotlib_gives_a_clear_error() -> None:
    code = (
        "import sys; sys.modules['matplotlib'] = None\n"
        "try:\n"
        "    import ertkit.plot\n"
        "except ImportError as e:\n"
        "    sys.exit(0 if 'ertkit[plot]' in str(e) else 2)\n"
        "sys.exit(1)\n"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
