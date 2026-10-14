"""Smooth (Occam) inversion and 1-D depth of investigation (design, 3.3 and 3.5)."""

from __future__ import annotations

import numpy as np
import pytest
from verification_report import Verification

from ertkit import ConvergenceWarning, DataWarning, ErrorModel
from ertkit._linalg import matmul
from ertkit.ves import LayeredModel, Sounding, Survey, forward
from ertkit.ves.smooth import (
    DOI_THRESHOLD,
    TARGET_UNREACHABLE,
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
        "Occam: |final chi2 - N_d| / sqrt(2 N_d)",
        abs(fit.chi2 - fit.n_data) / np.sqrt(2 * fit.n_data),
        1.0,
        "the final misfit must lie within one standard deviation of chi-squared "
        "with N_d degrees of freedom of its target N_d",
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


@pytest.mark.verification
def test_depth_of_investigation_index(
    verification: Verification, fit: SmoothFit
) -> None:
    tops = np.concatenate([[0.0], fit.model.depths])
    verification.check(
        "DOI: largest rescaled index of the layers above the depth of investigation",
        float(np.max(fit.doi_index[tops < fit.doi])),
        DOI_THRESHOLD,
        "every layer above the depth of investigation must be controlled by the data "
        "(Oldenburg and Li 1999)",
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
    # threshold near 1.6 m; the depth of investigation is where resolution is
    # lost, about 19 m, not at that fluctuation.
    survey = Survey.schlumberger(ab2=np.geomspace(2.5, 50.0, 24), mn2=0.5)
    result = invert_smooth(Sounding.synthetic(BASEMENT, survey, NOISE, rng=4))
    tops = np.concatenate([[0.0], result.model.depths])
    assert np.any((result.doi_index >= DOI_THRESHOLD) & (tops < 5.0))
    assert 10.0 < result.doi < 50.0


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
def test_depth_of_investigation_is_zero_when_nothing_is_resolved() -> None:
    # Errors of 200% leave every layer to the reference models.
    result = invert_smooth(_half_space_data(2.0))
    assert result.doi == 0.0
    assert np.min(result.doi_index) > DOI_THRESHOLD


def test_unreachable_target_returns_the_minimized_misfit() -> None:
    # 10% noise reported as 1%: chi2 = N_d cannot be reached.
    survey = Survey.schlumberger(ab2=np.geomspace(1.0, 300.0, 24), mn2=0.5)
    eps = np.random.default_rng(0).standard_normal(24)
    rhoa = forward(BASEMENT, survey) * np.exp(0.10 * eps)
    data = Sounding(survey=survey, rhoa=rhoa, error=ErrorModel(relative=0.01))
    with pytest.warns(DataWarning, match="underestimated"):
        result = invert_smooth(data, depth_of_investigation=False)
    assert result.converged
    assert result.stop_reason == TARGET_UNREACHABLE
    start = _Occam(data, result.model.thickness, 0.0, 0.0)
    uniform = np.full(result.model.n_layers, float(np.mean(data.log_data)))
    assert result.chi2 < start.chi2(uniform)


@pytest.mark.filterwarnings(
    "ignore:the target misfit cannot be reached:ertkit.DataWarning"
)
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
