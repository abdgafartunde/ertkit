"""Few-layer inversion (design document, sections 3.3 and 3.5)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats
from verification_report import Verification

from ertkit import ConvergenceWarning, DataWarning, ErrorModel
from ertkit.ves import LayeredModel, Sounding, Survey, forward
from ertkit.ves import invert as invert_module
from ertkit.ves.invert import LayerFit, invert_few_layer

NOISE = ErrorModel(relative=0.03)

# Stored noise vectors (standard normal draws), so that the recovery tests do
# not depend on the random-number stream of a particular numpy build.
EPS_DESIGN = np.array(
    [0.0342, 1.3597, 1.2247, -0.5103, -0.298, -0.5274, 0.5697, -0.0561, 0.7469,
     -1.8473, 1.5665, -0.0964, 0.6804, -0.1366, -0.3791, 0.4631, 0.8245, -0.2025,
     -0.1528, 0.6857, -0.8703, -1.5144]
)  # fmt: skip
EPS_BASEMENT = np.array(
    [-0.0068, 1.0461, 0.7416, 0.724, 1.6188, -1.2056, -0.627, -1.3207, -0.1078,
     0.9988, -0.0219, 0.4959, -1.9108, 0.1471, -0.9069, 1.7754, 0.8868, 0.9493,
     -0.0579, 0.6129, 0.6579, -0.3444, -0.4974, -0.1148, -0.6055, -0.5943]
)  # fmt: skip

DESIGN_MODEL = LayeredModel(resistivity=[150.0, 30.0, 800.0], thickness=[2.0, 15.0])
DESIGN_SURVEY = Survey.schlumberger(ab2=np.geomspace(2.5, 150.0, 22), mn2=0.5)

# Weathered basement typical of Nigeria's basement complex: topsoil, a
# clay-rich weathered layer, fresh crystalline basement (an H-type curve).
BASEMENT_MODEL = LayeredModel(resistivity=[200.0, 30.0, 2000.0], thickness=[1.5, 12.0])
_AB2 = np.geomspace(1.0, 300.0, 26)
BASEMENT_SURVEY = Survey.schlumberger(ab2=_AB2, mn2=np.where(_AB2 < 15, 0.5, 5.0))


def _noisy(model: LayeredModel, survey: Survey, eps: np.ndarray) -> Sounding:
    rhoa = forward(model, survey) * np.exp(NOISE.relative * eps)
    return Sounding(survey=survey, rhoa=rhoa, error=NOISE)


def _log_params(model: LayeredModel) -> np.ndarray:
    return np.log(np.concatenate([model.resistivity, model.thickness]))


@pytest.mark.verification
@pytest.mark.parametrize(
    ("name", "model", "survey", "eps"),
    [
        ("three-layer design example", DESIGN_MODEL, DESIGN_SURVEY, EPS_DESIGN),
        ("weathered basement", BASEMENT_MODEL, BASEMENT_SURVEY, EPS_BASEMENT),
    ],
)
def test_recovery_within_reported_uncertainty(
    verification: Verification,
    name: str,
    model: LayeredModel,
    survey: Survey,
    eps: np.ndarray,
) -> None:
    fit = invert_few_layer(_noisy(model, survey, eps), model.n_layers, rng=0)
    assert fit.converged
    error = _log_params(fit.model) - _log_params(model)
    p = error.size
    verification.check(
        f"{name}: largest parameter error in standard deviations",
        float(np.max(np.abs(error / fit.std_log))),
        3.0,
        "each log parameter should lie within 3 linearized standard deviations",
    )
    verification.check(
        f"{name}: Mahalanobis distance of the estimate from the truth",
        float(error @ np.linalg.solve(fit.covariance, error)),
        float(stats.chi2.ppf(0.997, p)),
        f"the 99.7% quantile of chi-squared with {p} degrees of freedom",
    )


@pytest.mark.slow
@pytest.mark.verification
def test_honest_uncertainty(verification: Verification) -> None:
    # A well-resolved model (all linearized standard deviations of the log
    # parameters below 0.2) with the true number of layers; the fits start
    # from the truth so that the test measures the estimator, not the search.
    model = DESIGN_MODEL
    survey = Survey.schlumberger(ab2=np.geomspace(1.0, 400.0, 28), mn2=0.5)
    n_draws, p = 1000, 5
    dof = survey.n_readings - p
    # With this seed the mean chi2 is 2.7 standard errors below N - p, inside
    # the 3-standard-error band. It is not a bias: 3000 draws with another seed
    # give 22.97 +/- 0.12 for N - p = 23.
    rng = np.random.default_rng(20261014)
    truth = _log_params(model)
    chi2, inside, mahalanobis, worst_std = [], [], [], 0.0
    for _ in range(n_draws):
        # About 1% of draws legitimately give p < 0.01, which warns.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DataWarning)
            fit = invert_few_layer(
                Sounding.synthetic(model, survey, NOISE, rng=rng), 3, start=model
            )
        error = _log_params(fit.model) - truth
        chi2.append(fit.chi2)
        inside.append(np.abs(error) <= fit.std_log)
        mahalanobis.append(error @ np.linalg.solve(fit.covariance, error))
        worst_std = max(worst_std, float(fit.std_log.max()))
    assert worst_std < 0.2, "the test model must be well resolved"
    band = 3 * np.sqrt(0.683 * 0.317 / n_draws)
    verification.check(
        "honest uncertainty: |mean chi2 - (N - p)|",
        abs(np.mean(chi2) - dof),
        3 * np.sqrt(2 * dof / n_draws),
        "three standard errors of the mean of a chi-squared variable with N - p "
        "degrees of freedom",
    )
    verification.check(
        "honest uncertainty: largest |coverage of 1-sd intervals - 0.683| over parameters",
        float(np.max(np.abs(np.mean(inside, axis=0) - 0.683))),
        band,
        "three binomial standard errors for 1000 draws",
    )
    verification.check(
        "honest uncertainty: |joint coverage of the 68.3% region - 0.683|",
        abs(float(np.mean(np.array(mahalanobis) <= stats.chi2.ppf(0.683, p))) - 0.683),
        band,
        "three binomial standard errors for 1000 draws",
    )


def test_equivalence_is_reported_as_large_uncertainty() -> None:
    # A fractured zone (400 ohm m) between a weathered layer and fresh basement
    # is suppressed: equivalent models fit as well, and the fit must not claim
    # a well-determined thickness for it.
    model = LayeredModel(
        resistivity=[150.0, 40.0, 400.0, 3000.0], thickness=[2.0, 15.0, 25.0]
    )
    survey = Survey.schlumberger(ab2=np.geomspace(1.5, 500.0, 30), mn2=0.5)
    eps = np.random.default_rng(3).standard_normal(survey.n_readings)
    fit = invert_few_layer(_noisy(model, survey, eps), 4, rng=0)
    assert fit.p_value > 0.01
    h3 = fit.parameter_names.index("h_3")
    assert fit.factor[h3] > 2.0


def test_static_shift_is_recovered() -> None:
    ab2 = np.concatenate([_AB2[_AB2 < 15], [12.0, 14.0], _AB2[_AB2 >= 15]])
    mn2 = np.where(np.arange(ab2.size) < np.sum(_AB2 < 15), 0.5, 5.0)
    survey = Survey.schlumberger(ab2=ab2, mn2=mn2)
    shift = np.where(mn2 > 1, 1.15, 1.0)
    eps = np.random.default_rng(5).standard_normal(ab2.size)
    data = Sounding(
        survey=survey,
        rhoa=forward(BASEMENT_MODEL, survey) * shift * np.exp(0.03 * eps),
        error=NOISE,
    )
    with pytest.warns(DataWarning, match="poor fit"):
        invert_few_layer(data, 3, rng=0)
    fit = invert_few_layer(data, 3, static_shift=True, rng=0)
    assert fit.parameter_names[-1] == "shift_2"
    assert fit.p_value > 0.01
    z = np.log(fit.static_shift[1] / 1.15) / fit.std_log[-1]
    assert abs(z) < 3


def test_start_model_gives_the_same_fit_as_the_search() -> None:
    data = _noisy(DESIGN_MODEL, DESIGN_SURVEY, EPS_DESIGN)
    searched = invert_few_layer(data, 3, rng=0)
    started = invert_few_layer(data, 3, start=DESIGN_MODEL)
    assert started.n_starts == 1
    np.testing.assert_allclose(started.chi2, searched.chi2, rtol=1e-8)
    np.testing.assert_allclose(
        started.model.resistivity, searched.model.resistivity, rtol=1e-5
    )


def test_result_contents() -> None:
    fit = invert_few_layer(_noisy(DESIGN_MODEL, DESIGN_SURVEY, EPS_DESIGN), 3, rng=0)
    assert isinstance(fit, LayerFit)
    assert fit.parameter_names == ("rho_1", "rho_2", "rho_3", "h_1", "h_2")
    assert fit.chi2_target == DESIGN_SURVEY.n_readings - 5
    np.testing.assert_allclose(fit.depths, np.cumsum(fit.model.thickness))
    np.testing.assert_allclose(np.diag(fit.correlation), 1.0)
    np.testing.assert_allclose(fit.covariance, fit.covariance.T)
    np.testing.assert_allclose(fit.factor, np.exp(fit.std_log))
    np.testing.assert_array_equal(fit.static_shift, [1.0])
    assert fit.predicted.shape == (DESIGN_SURVEY.n_readings,)
    assert fit.at_bound == ()
    assert 0.0 < fit.chi2_ratio < 3.0


def test_unresolved_layer_is_flagged() -> None:
    survey = Survey.schlumberger(ab2=np.geomspace(1.0, 300.0, 20), mn2=0.5)
    half_space = LayeredModel(resistivity=[100.0], thickness=[])
    data = Sounding.synthetic(half_space, survey, NOISE, rng=2)
    with pytest.warns(DataWarning, match="search bounds"):
        fit = invert_few_layer(data, 2, rng=0)
    assert fit.at_bound
    assert np.isinf(fit.factor).any() or fit.factor.max() > 1e3


# An unconverged fit is also a poor fit; that second warning is expected here.
@pytest.mark.filterwarnings("ignore::ertkit.DataWarning")
def test_warns_when_no_start_converges(monkeypatch: pytest.MonkeyPatch) -> None:
    real = invert_module.optimize.least_squares

    def one_step(*args: object, **kwargs: object) -> object:
        return real(*args, **{**kwargs, "max_nfev": 1})  # type: ignore[call-overload]

    monkeypatch.setattr(invert_module.optimize, "least_squares", one_step)
    data = _noisy(DESIGN_MODEL, DESIGN_SURVEY, EPS_DESIGN)
    with pytest.warns(ConvergenceWarning, match="no starting model converged"):
        fit = invert_few_layer(data, 3, n_starts=2, rng=0)
    assert not fit.converged


def test_poor_fit_warns() -> None:
    eps = np.random.default_rng(9).standard_normal(DESIGN_SURVEY.n_readings)
    rhoa = forward(DESIGN_MODEL, DESIGN_SURVEY) * np.exp(0.10 * eps)
    data = Sounding(survey=DESIGN_SURVEY, rhoa=rhoa, error=ErrorModel(relative=0.01))
    with pytest.warns(DataWarning, match="poor fit"):
        invert_few_layer(data, 3, rng=0)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_layers": 0}, "at least 1"),
        ({"n_layers": 12}, "cannot determine"),
        ({"n_layers": 2, "start": DESIGN_MODEL}, "start has 3 layers"),
    ],
)
def test_validation(kwargs: dict[str, object], match: str) -> None:
    data = _noisy(DESIGN_MODEL, DESIGN_SURVEY, EPS_DESIGN)
    with pytest.raises(ValueError, match=match):
        invert_few_layer(data, **kwargs)  # type: ignore[arg-type]
