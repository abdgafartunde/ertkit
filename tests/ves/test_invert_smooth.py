"""Smooth (Occam) inversion and 1-D depth of investigation (design, 3.3 and 3.5)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import optimize
from verification_report import Verification

from ertkit import ConvergenceWarning, DataWarning, ErrorModel
from ertkit._linalg import matmul
from ertkit.ves import LayeredModel, Sounding, Survey, forward
from ertkit.ves.smooth import (
    _REFIT_FACTOR,
    _TARGET_RELAXED,
    DOI_THRESHOLD,
    SmoothFit,
    _layer_thicknesses,
    _Occam,
    invert_smooth,
)

NOISE = ErrorModel(relative=0.03)
BASEMENT = LayeredModel(resistivity=[200.0, 30.0, 2000.0], thickness=[1.5, 12.0])


def _data(ab2_max: float, rng: int = 4) -> Sounding:
    survey = Survey.schlumberger(ab2=np.geomspace(1.0, ab2_max, 24), mn2=0.5)
    return Sounding.synthetic(BASEMENT, survey, NOISE, rng=rng)


@pytest.fixture(scope="module")
def fit() -> SmoothFit:
    return invert_smooth(_data(300.0))


@pytest.mark.verification
def test_occam_reaches_the_target_misfit(
    verification: Verification, fit: SmoothFit
) -> None:
    assert fit.converged
    verification.check(
        "Occam: |final chi2 - target| / target",
        abs(fit.chi2 - fit.chi2_target) / fit.chi2_target,
        0.01,
        "mu is found by root finding on chi2(mu) = target (N_d, or 1.1 times the "
        "smallest misfit when N_d is out of reach), so the final misfit must match "
        "it closely",
    )


@pytest.mark.verification
def test_occam_mu_is_the_largest_that_fits(
    verification: Verification, fit: SmoothFit
) -> None:
    data = _data(300.0)
    problem = _Occam(data, fit.model.thickness, alpha_s=0.0, m_ref=0.0)
    m = np.log(fit.model.resistivity)
    jac = problem.jacobian(m)
    wj = problem.w[:, np.newaxis] * jac
    gram = matmul(wj.T, wj)
    rhs = matmul(
        wj.T, problem.w * (problem.d - problem.predict(m).real + matmul(jac, m))
    )
    smoother = problem.chi2(problem.model_for(1.1 * fit.mu, gram, rhs))
    verification.check(
        "Occam: N_d - chi2 after raising mu by 10% (must be negative)",
        fit.n_data - smoother,
        0.0,
        "mu is the largest regularization weight with chi2 = N_d, so a smoother "
        "model must fit worse than the target",
    )


def test_smooth_model_follows_the_layers(fit: SmoothFit) -> None:
    tops = np.concatenate([[0.0], fit.model.depths])
    centers = tops + 0.5 * np.append(fit.model.thickness, fit.model.thickness[-1])
    # The conductive weathered layer (1.5 to 13.5 m) holds the smallest resistivity.
    z_min = centers[np.argmin(fit.model.resistivity)]
    assert 1.5 <= z_min <= 13.5
    assert fit.model.resistivity[0] > 100.0  # resistive topsoil
    assert fit.model.resistivity[-1] > 1000.0  # resistive basement


def _detectable_depth(model: LayeredModel, survey: Survey) -> float:
    """Depth below which a change of the earth would go unnoticed in the data.

    Multiplying or dividing every resistivity below depth z by 10^0.2 (0.2
    decades) changes the noise-free data by chi2 = 9 (three standard
    deviations, with 3% errors) at this depth, and by less below it. It is
    computed from the true model only, independently of the inversion.
    """
    base = np.log(forward(model, survey))
    tops = np.concatenate([[0.0], model.depths])

    def excess(log_z: float, factor: float) -> float:
        z = np.exp(log_z)
        k = int(np.searchsorted(tops, z, side="right"))  # layer k - 1 holds z
        depths = np.concatenate([model.depths[: k - 1], [z], model.depths[k - 1 :]])
        rho = np.concatenate(
            [model.resistivity[:k], model.resistivity[k - 1 :] * factor]
        )
        changed = LayeredModel(
            resistivity=rho, thickness=np.diff(np.concatenate([[0.0], depths]))
        )
        misfit = ((np.log(forward(changed, survey)) - base) / NOISE.relative) ** 2
        return float(np.sum(misfit)) - 9.0

    return max(
        float(np.exp(optimize.brentq(excess, np.log(1e-3), np.log(1e5), args=(f,))))
        for f in (10**0.2, 10**-0.2)
    )


@pytest.mark.verification
def test_depth_of_investigation_index(
    verification: Verification, fit: SmoothFit
) -> None:
    survey = Survey.schlumberger(ab2=np.geomspace(1.0, 300.0, 24), mn2=0.5)
    z_det = _detectable_depth(BASEMENT, survey)
    verification.check(
        "DOI: depth of investigation / detectable depth of the true model",
        fit.doi / z_det,
        1.25,
        "the data cannot see a 0.2-decade change below the detectable depth (chi2 "
        "change below 9), so a depth of investigation much deeper would claim "
        "resolution the data do not have",
    )
    verification.check(
        "DOI: detectable depth of the true model / depth of investigation",
        z_det / fit.doi,
        4.0,
        "a depth of investigation far shallower than the detectable depth would "
        "discard structure the data do resolve",
    )
    verification.check(
        "DOI: 1 - rescaled index of the second deepest layer",
        1.0 - float(fit.doi_index[-2]),
        0.1,
        "the deepest layers are controlled by the reference models (the bottom layer "
        "is 1 by construction, so the next one is checked)",
    )
    assert 0.0 < fit.doi < 300.0


def test_isolated_shallow_exceedance_does_not_set_the_depth() -> None:
    # With this survey one shallow layer's index fluctuates just above the
    # threshold near 1.6 m; the depth of investigation, found from the bottom
    # up, is where resolution is lost, about 19 m, not at that fluctuation.
    survey = Survey.schlumberger(ab2=np.geomspace(2.5, 50.0, 24), mn2=0.5)
    result = invert_smooth(Sounding.synthetic(BASEMENT, survey, NOISE, rng=4))
    tops = np.concatenate([[0.0], result.model.depths])
    assert np.any((result.doi_index >= DOI_THRESHOLD) & (tops < 5.0))
    assert 10.0 < result.doi < 50.0


README_MODEL = LayeredModel(resistivity=[150.0, 30.0, 800.0], thickness=[2.0, 15.0])


@pytest.mark.parametrize(
    ("survey", "seed"),
    [
        (Survey.schlumberger(ab2=np.geomspace(2.5, 200.0, 22), mn2=0.5), 205),
        (Survey.wenner(np.geomspace(1.0, 100.0, 20)), 102),
    ],
    ids=["schlumberger", "wenner"],
)
def test_overshoot_dips_do_not_make_the_depth_shallow(
    survey: Survey, seed: int
) -> None:
    # Before release 0.1 the first layer above the threshold set the depth:
    # for these noise draws an overshoot of the reference inversions near the
    # top gave depths of investigation of a few meters.
    data = Sounding.synthetic(README_MODEL, survey, NOISE, rng=seed)
    assert invert_smooth(data).doi > 10.0


def test_depth_of_investigation_hardly_depends_on_the_number_of_layers() -> None:
    survey = Survey.schlumberger(ab2=np.geomspace(2.5, 200.0, 22), mn2=0.5)
    data = Sounding.synthetic(README_MODEL, survey, NOISE, rng=4)
    depths = [invert_smooth(data, n_layers=n).doi for n in (20, 30, 45, 60)]
    assert max(depths) < 1.5 * min(depths)


def test_depth_of_investigation_grows_with_spread_length() -> None:
    depths = [invert_smooth(_data(ab2_max)).doi for ab2_max in (40.0, 300.0, 1000.0)]
    assert depths[0] < depths[1] < depths[2]


def test_sensitivity_decreases_with_depth(fit: SmoothFit) -> None:
    assert fit.sensitivity.shape == (30,)
    assert fit.sensitivity[:5].mean() > 10 * fit.sensitivity[-5:].mean()


def test_depth_of_investigation_can_be_skipped() -> None:
    result = invert_smooth(_data(100.0), depth_of_investigation=False)
    assert np.isnan(result.doi)
    assert np.isnan(result.doi_index).all()


def test_undetermined_depth_of_investigation_warns() -> None:
    # A model whose half-space starts at 30 m can fit the data, but a 300 m
    # spread resolves it to the bottom, so the reference models agree there.
    with pytest.warns(DataWarning, match="could not be determined"):
        result = invert_smooth(_data(300.0), depth_range=(0.3, 30.0))
    assert result.converged
    assert np.isnan(result.doi)


def _half_space_data(relative_error: float) -> Sounding:
    survey = Survey.schlumberger(ab2=np.geomspace(1.0, 100.0, 24), mn2=0.5)
    model = LayeredModel(resistivity=[100.0], thickness=[])
    eps = np.random.default_rng(1).standard_normal(24)
    rhoa = forward(model, survey) * np.exp(0.01 * eps)
    return Sounding(survey=survey, rhoa=rhoa, error=ErrorModel(relative=relative_error))


def test_overstated_errors_give_the_smoothest_model_and_warn() -> None:
    # 1% noise reported as 5%: even a uniform model fits better than chi2 = N_d.
    with pytest.warns(DataWarning, match="overestimated"):
        result = invert_smooth(_half_space_data(0.05), depth_of_investigation=False)
    assert result.converged
    assert result.chi2 < result.n_data
    np.testing.assert_allclose(result.model.resistivity, 100.0, rtol=0.02)
    assert np.ptp(result.model.resistivity) < 1e-3 * result.model.resistivity.mean()


@pytest.mark.filterwarnings("ignore:the smoothest model:ertkit.DataWarning")
@pytest.mark.filterwarnings("ignore:relative error 2 is 200%:ertkit.DataWarning")
def test_depth_of_investigation_is_zero_when_nothing_is_resolved() -> None:
    # Errors of 200% leave every layer to the reference models.
    result = invert_smooth(_half_space_data(2.0))
    assert result.doi == 0.0
    assert np.min(result.doi_index) > DOI_THRESHOLD


def test_unreachable_target_is_relaxed_and_a_poor_fit_reported() -> None:
    # 10% noise reported as 1%: chi2 = N_d cannot be reached, and even the
    # smallest misfit is improbably large.
    survey = Survey.schlumberger(ab2=np.geomspace(1.0, 300.0, 24), mn2=0.5)
    eps = np.random.default_rng(0).standard_normal(24)
    rhoa = forward(BASEMENT, survey) * np.exp(0.10 * eps)
    data = Sounding(survey=survey, rhoa=rhoa, error=ErrorModel(relative=0.01))
    with pytest.warns(DataWarning, match="underestimated"):
        result = invert_smooth(data, depth_of_investigation=False)
    assert result.converged
    assert result.stop_reason == _TARGET_RELAXED
    assert result.chi2_target > data.n_data
    assert result.chi2 == pytest.approx(result.chi2_target, rel=1e-3)
    start = _Occam(data, result.model.thickness, 0.0, 0.0)
    uniform = np.full(result.model.n_layers, float(np.mean(data.log_data)))
    assert result.chi2 < start.chi2(uniform)


def test_unreachable_target_with_correct_errors_is_not_a_poor_fit() -> None:
    # With correct errors chi2 = N_d is out of reach for a few percent of
    # soundings (this seed); the smallest misfit is plausible, so the
    # inversion fits 1.1 times it without a warning.
    data = _data(300.0, rng=108)
    result = invert_smooth(data)
    assert result.converged
    assert result.stop_reason == _TARGET_RELAXED
    chi2_min = result.chi2_target / _REFIT_FACTOR
    assert data.n_data < result.chi2_target < 1.5 * data.n_data
    assert chi2_min < result.chi2_target
    assert result.chi2 == pytest.approx(result.chi2_target, rel=1e-3)
    assert np.isfinite(result.doi)


@pytest.mark.filterwarnings("ignore:poor fit:ertkit.DataWarning")
def test_extreme_data_do_not_overflow() -> None:
    # A contrast of 1e4 with 20% noise reported as 0.5%, on 80 layers: before
    # the fix the model overflowed to nan.
    survey = Survey.schlumberger(ab2=np.geomspace(1.0, 300.0, 24), mn2=0.5)
    eps = np.random.default_rng(0).standard_normal(24)
    model = LayeredModel(resistivity=[1e4, 1.0], thickness=[5.0])
    rhoa = forward(model, survey) * np.exp(0.2 * eps)
    data = Sounding(survey=survey, rhoa=rhoa, error=ErrorModel(relative=0.005))
    result = invert_smooth(data, n_layers=80, depth_of_investigation=False)
    assert np.isfinite(result.chi2)
    assert np.all(np.isfinite(result.model.resistivity))


def test_occam_jacobian_rows_sum_to_one() -> None:
    # rho_a is homogeneous of degree one in the layer resistivities, so the
    # derivatives of ln rho_a with respect to every ln rho_j sum to 1.
    data = _data(300.0)
    problem = _Occam(data, _layer_thicknesses(data, 30, None), 0.0, 0.0)
    m = np.log(np.geomspace(50.0, 2000.0, 30))
    np.testing.assert_allclose(problem.jacobian(m).sum(axis=1), 1.0, atol=1e-10)


def test_layer_thicknesses() -> None:
    h = _layer_thicknesses(_data(300.0), 30, None)
    assert h.size == 29
    np.testing.assert_allclose(np.cumsum(h)[[0, -1]], [0.3, 900.0])


def test_too_few_iterations_warns() -> None:
    with pytest.warns(ConvergenceWarning, match="stopped after 1 iterations"):
        result = invert_smooth(_data(100.0), max_iter=1, depth_of_investigation=False)
    assert not result.converged


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_layers": 1}, "at least 2 layers"),
        ({"max_iter": 0}, "max_iter must be at least 1"),
        ({"depth_range": (5.0, 1.0)}, "0 < top < bottom"),
        ({"depth_range": (0.0, 10.0)}, "0 < top < bottom"),
    ],
)
def test_validation(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        invert_smooth(_data(100.0), **kwargs)  # type: ignore[arg-type]


def test_residuals_and_summary(fit: SmoothFit) -> None:
    assert np.sum(fit.residuals**2) == pytest.approx(fit.chi2, rel=1e-10)
    summary = fit.summary()
    assert summary.startswith("SmoothFit: chi2 = ")
    assert f"depth of investigation: {fit.doi:.3g} m" in summary


def test_data_must_be_a_sounding() -> None:
    with pytest.raises(TypeError, match="data must be a Sounding"):
        invert_smooth(np.ones(10))  # type: ignore[arg-type]


def test_poor_fit_with_several_mn_segments_suggests_static_shifts() -> None:
    ab2 = np.geomspace(1.0, 300.0, 24)
    survey = Survey.schlumberger(ab2=ab2, mn2=np.where(ab2 < 15, 0.5, 5.0))
    shift = np.where(ab2 < 15, 1.0, 1.4)
    eps = np.random.default_rng(0).standard_normal(24)
    rhoa = forward(BASEMENT, survey) * shift * np.exp(0.01 * eps)
    data = Sounding(survey=survey, rhoa=rhoa, error=ErrorModel(relative=0.01))
    with pytest.warns(DataWarning, match="static_shift=True"):
        invert_smooth(data, depth_of_investigation=False)
