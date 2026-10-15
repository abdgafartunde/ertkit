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


def test_wenner_fit_is_drawn_as_one_curve() -> None:
    # MN changes with every Wenner reading; one-reading segments would be
    # invisible single points.
    survey = Survey.wenner(a=np.geomspace(1.0, 150.0, 18))
    wenner = Sounding.synthetic(MODEL, survey, rng=2)
    fit = invert_few_layer(wenner, 3, rng=0)
    ax = plot.sounding(wenner, fit)
    assert max(len(line.get_xdata()) for line in ax.get_lines()) == survey.n_readings


@pytest.mark.filterwarnings("ignore::ertkit.DataWarning")
def test_unbounded_error_bars_stay_on_the_plot() -> None:
    # A half-space fitted with two layers: the interface depth and the bottom
    # resistivity are undetermined.
    survey = Survey.schlumberger(ab2=np.geomspace(1.0, 300.0, 20), mn2=0.5)
    half_space = LayeredModel(resistivity=[100.0], thickness=[])
    fit = invert_few_layer(Sounding.synthetic(half_space, survey, rng=2), 2, rng=0)
    assert np.isinf(fit.factor).any() or fit.factor.max() > 1e3
    ax = plot.layered_model(fit, depth_max=50.0)
    for collection in ax.collections:
        for segment in collection.get_segments():
            assert np.all(np.isfinite(segment))
    assert ax.get_ylim() == (50.0, 0.0)
    left, right = ax.get_xlim()
    assert right / left < 1e6


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
