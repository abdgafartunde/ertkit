"""Verification of the VES forward model (design document, section 3.5)."""

from __future__ import annotations

import numpy as np
import pytest
from analytic_solutions import lipschitz_integral, two_layer_potential
from hypothesis import given
from hypothesis import strategies as st
from verification_report import Verification

from ertkit.ves import LayeredModel, Survey, forward, hankel_j0, potential
from ertkit.ves.forward import _rhoa, _tanh

CONTRASTS = [1e-3, 1e-2, 1e-1, 1e1, 1e2, 1e3]  # rho2 / rho1, capped at 1000
R_OVER_H = np.geomspace(0.1, 100.0, 41)


@pytest.mark.verification
def test_filter_on_lipschitz_integral(verification: Verification) -> None:
    r = np.geomspace(0.05, 5000.0, 200)
    worst = 0.0
    for a in [0.01, 0.1, 1.0, 10.0, 100.0]:
        approx = hankel_j0(lambda lam, a=a: np.exp(-a * lam), r)
        worst = max(worst, float(np.max(np.abs(approx / lipschitz_integral(a, r) - 1))))
    verification.check(
        "max relative error of the J0 filter on the Lipschitz integral (a/r up to 2000)",
        worst,
        1e-8,
        "the layered-earth kernel is a sum of such terms; 1e-8 leaves two orders of "
        "magnitude for the 1e-6 gate on two-layer models",
    )


@pytest.mark.verification
def test_half_space_is_exact(verification: Verification) -> None:
    ab2 = np.geomspace(1.0, 1000.0, 25)
    model = LayeredModel(resistivity=[42.0], thickness=[])
    worst = 0.0
    for survey in (Survey.schlumberger(ab2, ab2 / 20), Survey.wenner(ab2)):
        worst = max(worst, float(np.max(np.abs(forward(model, survey) / 42.0 - 1))))
    verification.check(
        "max relative error of apparent resistivity over a half-space",
        worst,
        1e-6,
        "computed without subtracting the half-space, so this tests that the filter "
        "weights integrate a constant kernel; the design gate is 1e-6",
    )


@pytest.mark.verification
def test_two_layer_potential_against_image_series(verification: Verification) -> None:
    h, rho1 = 5.0, 100.0
    r = R_OVER_H * h
    worst = 0.0
    for contrast in CONTRASTS:
        model = LayeredModel(resistivity=[rho1, contrast * rho1], thickness=[h])
        exact = two_layer_potential(r, rho1, contrast * rho1, h)
        worst = max(worst, float(np.max(np.abs(potential(model, r) / exact - 1))))
    verification.check(
        "max relative error in V, two-layer earths, contrast 1e-3 to 1e3, r/h 0.1 to 100",
        worst,
        1e-6,
        "design gate of section 3.5: well below any measurement error",
    )


@pytest.mark.verification
@pytest.mark.parametrize("mn_ratio", [0.2, 0.02])
def test_two_layer_schlumberger_against_image_series(
    verification: Verification, mn_ratio: float
) -> None:
    h, rho1 = 5.0, 100.0
    ab2 = R_OVER_H * h
    mn2 = mn_ratio * ab2
    survey = Survey.schlumberger(ab2, mn2)
    worst = 0.0
    for contrast in CONTRASTS:
        rho2 = contrast * rho1
        model = LayeredModel(resistivity=[rho1, rho2], thickness=[h])
        v_near = two_layer_potential(ab2 - mn2, rho1, rho2, h)
        v_far = two_layer_potential(ab2 + mn2, rho1, rho2, h)
        exact = survey.geometric_factor * 2.0 * (v_near - v_far)
        worst = max(worst, float(np.max(np.abs(forward(model, survey) / exact - 1))))
    verification.check(
        f"max relative error in Schlumberger rho_a with MN/AB = {mn_ratio}",
        worst,
        1e-6,
        "design gate of section 3.5; differencing the potentials must not amplify "
        "the filter error",
    )


@pytest.mark.verification
def test_limits_at_small_and_large_spacing(verification: Verification) -> None:
    model = LayeredModel(resistivity=[100.0, 50.0, 200.0], thickness=[2.0, 10.0])
    depth = float(model.depths[-1])
    survey = Survey.schlumberger(
        ab2=[0.01 * 2.0, 1000.0 * depth], mn2=[0.002, 0.1 * depth]
    )
    rhoa = forward(model, survey)
    verification.check(
        "relative deviation from rho_1 at AB/2 = h_1/100",
        abs(rhoa[0] / 100.0 - 1),
        1e-6,
        "the deviation decays faster than linearly as AB/2 shrinks; it is about 1e-7 here",
    )
    verification.check(
        "relative deviation from rho_N at AB/2 = 1000 times the depth to the half-space",
        abs(rhoa[1] / 200.0 - 1),
        1e-4,
        "the deviation decays like (depth / AB2)^2, about 1e-5 at this spacing",
    )


@pytest.mark.verification
def test_complex_step_matches_finite_differences(verification: Verification) -> None:
    rho = np.array([150.0, 30.0, 800.0])
    h = np.array([2.0, 15.0])
    ab2 = np.geomspace(2.5, 150.0, 12)
    mn2 = np.full_like(ab2, 0.5)
    params = np.concatenate([rho, h])

    def rhoa(p: np.ndarray) -> np.ndarray:
        return _rhoa(p[: rho.size], p[rho.size :], ab2, mn2)

    worst = 0.0
    for i in range(params.size):
        bumped = params.astype(complex)
        bumped[i] += 1e-30j
        d_complex = rhoa(bumped).imag / 1e-30
        # Relative step 1e-4: below it rounding error dominates the central
        # difference, above it truncation error does.
        eps = 1e-4 * params[i]
        up, down = params.copy(), params.copy()
        up[i] += eps
        down[i] -= eps
        d_central = (rhoa(up).real - rhoa(down).real) / (2 * eps)
        worst = max(worst, float(np.max(np.abs(d_central / d_complex - 1))))
    verification.check(
        "max relative difference between complex-step and central-difference "
        "derivatives of rho_a with respect to every resistivity and thickness",
        worst,
        1e-6,
        "central differences at their optimal step are accurate to about 1e-7; the "
        "complex-step derivative is exact to round-off",
    )


def test_complex_tanh_is_finite_for_large_arguments() -> None:
    # numpy's complex tanh overflows on Windows; the transform must not.
    z = np.geomspace(1e-9, 1e5, 500) + 1e-30j
    t = _tanh(z)
    assert np.all(np.isfinite(t))
    np.testing.assert_allclose(t.real, np.tanh(z.real), rtol=1e-15)
    np.testing.assert_allclose(
        t.imag / 1e-30, 1 / np.cosh(np.minimum(z.real, 350)) ** 2, atol=1e-15
    )


def test_potential_rejects_bad_distances() -> None:
    model = LayeredModel(resistivity=[10.0], thickness=[])
    with pytest.raises(ValueError, match="positive"):
        potential(model, [1.0, 0.0])


def test_potential_scales_with_current() -> None:
    model = LayeredModel(resistivity=[10.0, 100.0], thickness=[3.0])
    np.testing.assert_allclose(
        potential(model, [5.0], current=2.0), 2 * potential(model, [5.0])
    )


# Invariants, checked with Hypothesis on random layered models and soundings.

layer_rho = st.floats(min_value=1.0, max_value=1e4)
layer_h = st.floats(min_value=0.1, max_value=100.0)


@st.composite
def models(draw: st.DrawFn) -> LayeredModel:
    n = draw(st.integers(min_value=1, max_value=5))
    rho = draw(st.lists(layer_rho, min_size=n, max_size=n))
    h = draw(st.lists(layer_h, min_size=n - 1, max_size=n - 1))
    return LayeredModel(resistivity=rho, thickness=h)


@st.composite
def surveys(draw: st.DrawFn) -> Survey:
    ab2 = np.array(sorted(draw(st.lists(st.floats(1.0, 1e3), min_size=1, max_size=8))))
    ratio = draw(st.floats(min_value=0.01, max_value=0.5))
    return Survey.schlumberger(ab2=ab2, mn2=ratio * ab2)


@given(models(), surveys(), st.floats(min_value=0.01, max_value=100.0))
def test_rhoa_is_homogeneous_in_resistivity(
    model: LayeredModel, survey: Survey, c: float
) -> None:
    scaled = LayeredModel(resistivity=c * model.resistivity, thickness=model.thickness)
    np.testing.assert_allclose(
        forward(scaled, survey), c * forward(model, survey), rtol=1e-10
    )


@given(models(), surveys(), st.floats(min_value=0.1, max_value=10.0))
def test_rhoa_is_invariant_under_length_scaling(
    model: LayeredModel, survey: Survey, c: float
) -> None:
    scaled_model = LayeredModel(
        resistivity=model.resistivity, thickness=c * model.thickness
    )
    scaled_survey = Survey.schlumberger(ab2=c * survey.ab2, mn2=c * survey.mn2)
    np.testing.assert_allclose(
        forward(scaled_model, scaled_survey), forward(model, survey), rtol=1e-10
    )


@given(models(), surveys())
def test_rhoa_is_positive(model: LayeredModel, survey: Survey) -> None:
    assert np.all(forward(model, survey) > 0)
