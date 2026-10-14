"""Regression tests for the findings of the pre-release review of 0.1.

Each test reproduces a defect found by the review (multi-start selection,
the linearized covariance, static-shift segments, the few-layer recovery of a
four-layer model) and checks that it stays fixed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from verification_report import Verification

from ertkit import ErrorModel
from ertkit.ves import LayeredModel, Sounding, Survey, forward, invert_few_layer
from ertkit.ves import invert as invert_module
from ertkit.ves.invert import _linearized_covariance, _Problem, _segments

NOISE = ErrorModel(relative=0.03)

# Stored standard normal draws (see tests/ves/test_invert_few_layer.py).
EPS_K_TYPE = np.array(
    [1.1033, -1.2798, 0.6482, -1.1996, 1.0718, -1.5561, -0.7105, -0.6067, -2.1053,
     0.8528, -1.5514, -1.3073, -0.5647, 0.7442, -0.8133, 1.9888, -0.0249, -0.2891,
     0.4486, -0.1312, -1.0577, -0.5499, 0.1667, 1.577, 0.2634, 0.4755, -2.2375,
     -0.2678, -0.0279, 1.7597]
)  # fmt: skip
EPS_FRACTURED = np.array(
    [-0.6518, -0.1747, 1.6637, 0.6591, -1.6414, -0.0052, -0.6235, 0.1486, -1.6082,
     0.2418, 0.2354, 1.5756, 0.3166, 0.5105, -1.4931, 2.2527, -1.9156, 1.1018,
     -0.3299, -0.8806, -0.6563, -0.672, 0.3802, -0.1101, 1.4826, -1.8296, -0.0031,
     -0.8921, 0.7759, -2.1181]
)  # fmt: skip
EPS_KH = np.array(
    [0.1257, -0.1321, 0.6404, 0.1049, -0.5357, 0.3616, 1.304, 0.9471, -0.7037,
     -1.2654, -0.6233, 0.0413, -2.325, -0.2188, -1.2459, -0.7323, -0.5443, -0.3163,
     0.4116, 1.0425, -0.1285, 1.3665, -0.6652, 0.3515, 0.9035, 0.094, -0.7435,
     -0.9217, -0.4577, 0.2202, -1.0096, -0.2092]
)  # fmt: skip


def _data(model: LayeredModel, survey: Survey, eps: np.ndarray) -> Sounding:
    rhoa = forward(model, survey) * np.exp(NOISE.relative * eps)
    return Sounding(survey=survey, rhoa=rhoa, error=NOISE)


def test_the_start_with_the_smallest_misfit_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A K-type model: most random starts reach the same minimum along an
    # equivalence ridge; the result must be the start with the smallest cost,
    # not one that merely met its convergence test.
    model = LayeredModel(
        resistivity=[50.0, 500.0, 20.0, 1000.0], thickness=[3.0, 10.0, 30.0]
    )
    data = _data(
        model,
        Survey.schlumberger(ab2=np.geomspace(1.0, 500.0, 30), mn2=0.5),
        EPS_K_TYPE,
    )
    costs: list[float] = []
    real = invert_module.optimize.least_squares

    def spy(*args: object, **kwargs: object) -> object:
        result = real(*args, **kwargs)  # type: ignore[call-overload]
        costs.append(float(result.cost))
        return result

    monkeypatch.setattr(invert_module.optimize, "least_squares", spy)
    fit = invert_few_layer(data, 4, rng=3)
    assert fit.chi2 == pytest.approx(2 * min(costs), rel=1e-12)
    assert fit.converged
    assert fit.p_value > 0.01


@pytest.mark.filterwarnings(
    "ignore:parameters at their search bounds:ertkit.DataWarning"
)
def test_unresolved_parameters_have_infinite_or_huge_uncertainty() -> None:
    # A suppressed fractured zone with a short-spacing gap: the topsoil is
    # unresolved. Before the fix the covariance was formed as (J^T W^2 J) and
    # pseudo-inverted, which discarded the null direction and reported factors
    # near 1.3 for these parameters.
    model = LayeredModel(
        resistivity=[150.0, 40.0, 400.0, 3000.0], thickness=[2.0, 15.0, 25.0]
    )
    survey = Survey.schlumberger(ab2=np.geomspace(1.5, 500.0, 30), mn2=0.5)
    fit = invert_few_layer(_data(model, survey, EPS_FRACTURED), 4, rng=0)
    factor = dict(zip(fit.parameter_names, fit.factor, strict=True))
    assert np.isinf(factor["rho_1"])
    assert factor["h_2"] > 10.0
    assert np.all(fit.depth_std > 100.0)


def test_exactly_degenerate_layers_are_not_reported_as_resolved() -> None:
    # Two layers with the same resistivity: only their total thickness matters,
    # so at this model the individual thicknesses are undetermined.
    model = LayeredModel(resistivity=[100.0, 100.0, 500.0], thickness=[3.0, 7.0])
    survey = Survey.schlumberger(ab2=np.geomspace(1.0, 300.0, 25), mn2=0.5)
    problem = _Problem(_data(model, survey, EPS_K_TYPE[:25]), 3, static_shift=False)
    m = np.log(np.concatenate([model.resistivity, model.thickness]))
    _, std, _ = _linearized_covariance(problem.jacobian(m) / problem.s[:, np.newaxis])
    assert np.all(np.isinf(std[3:]))  # h_1 and h_2


def test_linearized_covariance() -> None:
    rng = np.random.default_rng(0)
    full = rng.standard_normal((20, 4))
    cov, std, corr = _linearized_covariance(full)
    np.testing.assert_allclose(cov, np.linalg.inv(full.T @ full), rtol=1e-10)
    np.testing.assert_allclose(np.diag(corr), 1.0)
    # An exact null direction (columns 2 and 3 enter only through their sum).
    degenerate = np.column_stack([full[:, :2], full[:, 2], full[:, 2]])
    cov, std, corr = _linearized_covariance(degenerate)
    assert np.all(np.isinf(std[2:]))
    assert np.all(np.isfinite(std[:2]))
    assert np.isnan(corr[0, 2])
    assert corr[0, 0] == 1.0


def test_segments_are_grouped_by_mn_value() -> None:
    # Rows sorted by AB/2, with overlap readings interleaved between segments.
    mn2 = np.array([0.5, 0.5, 2.0, 0.5, 2.0, 2.0, 0.5, 2.0])
    np.testing.assert_array_equal(_segments(mn2), [0, 0, 1, 0, 1, 1, 0, 1])


def test_static_shift_with_interleaved_rows() -> None:
    model = LayeredModel(resistivity=[200.0, 30.0, 2000.0], thickness=[1.5, 12.0])
    short = np.geomspace(1.0, 14.0, 10)
    long = np.geomspace(8.0, 300.0, 16)
    ab2 = np.concatenate([short, long])
    mn2 = np.concatenate([np.full(short.size, 0.5), np.full(long.size, 2.0)])
    order = np.argsort(ab2)  # as a field sheet sorted by AB/2
    survey = Survey.schlumberger(ab2=ab2[order], mn2=mn2[order])
    shift = np.where(survey.mn2 > 1, 1.15, 1.0)
    rhoa = forward(model, survey) * shift * np.exp(0.03 * EPS_KH[:26])
    fit = invert_few_layer(
        Sounding(survey=survey, rhoa=rhoa, error=NOISE), 3, static_shift=True, rng=0
    )
    assert fit.parameter_names[-1] == "shift_2"
    assert sum(name.startswith("shift") for name in fit.parameter_names) == 1
    assert abs(np.log(fit.static_shift[1] / 1.15)) < 3 * fit.std_log[-1]


def test_jacobian_of_the_log_parameters() -> None:
    # The Jacobian used by the inversion (log parameters and a shift column)
    # against central differences at their optimal step.
    model = LayeredModel(resistivity=[150.0, 30.0, 800.0], thickness=[2.0, 15.0])
    survey = Survey.schlumberger(
        ab2=np.geomspace(2.5, 200.0, 20),
        mn2=np.where(np.geomspace(2.5, 200.0, 20) < 25, 0.5, 5.0),
    )
    problem = _Problem(_data(model, survey, EPS_KH[:20]), 3, static_shift=True)
    m = np.concatenate([np.log(model.resistivity), np.log(model.thickness), [0.1]])
    jac = problem.jacobian(m)
    for k in range(m.size):
        up, down = m.copy(), m.copy()
        up[k] += 1e-5
        down[k] -= 1e-5
        central = (problem.predict(up).real - problem.predict(down).real) / 2e-5
        np.testing.assert_allclose(jac[:, k], central, rtol=1e-6, atol=1e-9)


@pytest.mark.verification
def test_recovery_of_a_four_layer_model(verification: Verification) -> None:
    # A KH-type profile that the survey resolves (all linearized standard
    # deviations of the log parameters are small): topsoil, a resistive
    # layer, a conductive weathered zone and fresh basement.
    model = LayeredModel(
        resistivity=[100.0, 600.0, 25.0, 3000.0], thickness=[1.5, 4.0, 15.0]
    )
    ab2 = np.geomspace(1.0, 500.0, 32)
    mn2 = np.select([ab2 < 10, ab2 < 100], [0.2, 2.0], default=20.0)
    fit = invert_few_layer(
        _data(model, Survey.schlumberger(ab2=ab2, mn2=mn2), EPS_KH), 4, rng=0
    )
    assert fit.converged
    error = np.log(
        np.concatenate([fit.model.resistivity, fit.model.thickness])
    ) - np.log(np.concatenate([model.resistivity, model.thickness]))
    verification.check(
        "four-layer KH model: largest parameter error in standard deviations",
        float(np.max(np.abs(error / fit.std_log))),
        3.0,
        "each log parameter should lie within 3 linearized standard deviations",
    )
    verification.check(
        "four-layer KH model: Mahalanobis distance of the estimate from the truth",
        float(error @ np.linalg.solve(fit.covariance, error)),
        float(stats.chi2.ppf(0.997, error.size)),
        f"the 99.7% quantile of chi-squared with {error.size} degrees of freedom",
    )


@pytest.mark.slow
@pytest.mark.verification
@pytest.mark.filterwarnings("ignore::ertkit.DataWarning")
def test_coverage_under_equivalence_is_reported(verification: Verification) -> None:
    # Design 3.5, item 6: a model affected by equivalence reports its coverage
    # without a pass or fail threshold, to show why the sampler of release 0.2
    # is needed.
    model = LayeredModel(
        resistivity=[150.0, 40.0, 400.0, 3000.0], thickness=[2.0, 15.0, 25.0]
    )
    ab2 = np.geomspace(1.0, 500.0, 32)
    survey = Survey.schlumberger(
        ab2=ab2, mn2=np.select([ab2 < 10, ab2 < 100], [0.2, 2.0], default=20.0)
    )
    truth = np.log(np.concatenate([model.resistivity, model.thickness]))
    rng = np.random.default_rng(20261015)
    inside = []
    for _ in range(300):
        fit = invert_few_layer(
            Sounding.synthetic(model, survey, NOISE, rng=rng), 4, start=model
        )
        error = (
            np.log(np.concatenate([fit.model.resistivity, fit.model.thickness])) - truth
        )
        inside.append(np.abs(error) <= fit.std_log)
    coverage = np.mean(inside, axis=0)
    verification.check(
        "equivalence-affected four-layer model: lowest coverage of 1-sd intervals "
        "(reported only; 0.683 if the linearization held)",
        float(coverage.min()),
        float("inf"),
        "reported without a threshold (design 3.5, item 6): linearized intervals are "
        "not reliable under equivalence",
    )
