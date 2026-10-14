"""Cross-checks of the VES forward model against pyGIMLi and SimPEG.

Design document, section 3.5, item 9. These tests run in the weekly
cross-check job, which installs the codes in ``ci/crosscheck.txt`` and sets
``ERTKIT_REQUIRE_CROSSCHECK=1`` so that a missing package fails instead of
skipping. Elsewhere they are skipped.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

import numpy as np
import pytest
from verification_report import Verification

from ertkit.ves import LayeredModel, Survey, forward

# Warnings raised inside the other packages are not ertkit's concern here.
pytestmark = [
    pytest.mark.crosscheck,
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning:pygimli"),
]

MODELS = {
    "three-layer": LayeredModel(
        resistivity=[150.0, 30.0, 800.0], thickness=[2.0, 15.0]
    ),
    "weathered basement": LayeredModel(
        resistivity=[200.0, 30.0, 2000.0], thickness=[1.5, 12.0]
    ),
    "fractured basement": LayeredModel(
        resistivity=[150.0, 40.0, 400.0, 3000.0], thickness=[2.0, 15.0, 25.0]
    ),
    "contrast 1000": LayeredModel(resistivity=[1000.0, 1.0], thickness=[5.0]),
}
AB2 = np.geomspace(2.5, 300.0, 20)
MN2 = np.where(AB2 < 25, 0.5, 5.0)
SURVEY = Survey.schlumberger(ab2=AB2, mn2=MN2)


def _require(name: str) -> ModuleType:
    if os.environ.get("ERTKIT_REQUIRE_CROSSCHECK") == "1":
        return importlib.import_module(name)
    return pytest.importorskip(name, exc_type=ImportError)


@pytest.mark.parametrize("name", list(MODELS))
def test_against_pygimli(verification: Verification, name: str) -> None:
    ves = _require("pygimli.physics.ves")
    model = MODELS[name]
    fop = ves.VESModelling(ab2=AB2, mn2=MN2)
    theirs = np.asarray(
        fop.response(np.concatenate([model.thickness, model.resistivity]))
    )
    ours = forward(model, SURVEY)
    verification.check(
        f"pyGIMLi VES modeling, {name}: max relative difference in rho_a",
        float(np.max(np.abs(theirs / ours - 1))),
        1e-3,
        "design gate of 0.1%: two independent codes with different Hankel filters",
    )


@pytest.mark.parametrize("name", list(MODELS))
def test_against_simpeg(verification: Verification, name: str) -> None:
    dc = _require("simpeg.electromagnetics.static.resistivity")
    model = MODELS[name]
    sources = []
    for a, m in zip(AB2, MN2, strict=True):
        rx = dc.receivers.Dipole(
            locations_m=np.array([[-m, 0.0, 0.0]]),
            locations_n=np.array([[m, 0.0, 0.0]]),
        )
        sources.append(
            dc.sources.Dipole(
                [rx],
                location_a=np.array([-a, 0.0, 0.0]),
                location_b=np.array([a, 0.0, 0.0]),
            )
        )
    simulation = dc.Simulation1DLayers(
        survey=dc.Survey(sources),
        rho=model.resistivity,
        thicknesses=model.thickness,
        # A filter different from ertkit's, so that the comparison is independent.
        hankel_filter="anderson_801_1982",
    )
    theirs = SURVEY.geometric_factor * np.asarray(simulation.dpred())
    ours = forward(model, SURVEY)
    verification.check(
        f"SimPEG (Anderson 801-point filter), {name}: max relative difference in rho_a",
        float(np.max(np.abs(theirs / ours - 1))),
        1e-5,
        "both filters are accurate to about 1e-6 on these models",
    )
