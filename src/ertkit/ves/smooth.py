r"""Smooth (Occam) inversion of VES data, with a 1-D depth of investigation.

The model has many thin layers with fixed thicknesses that grow
logarithmically with depth; the unknowns are :math:`m = \ln \rho`. Each
iteration of Occam's inversion (Constable et al. 1987) solves for the model
itself, after linearizing about the current model :math:`m_k`:

.. math::

    m(\mu) = \bigl[\mu R + (WJ)^T WJ\bigr]^{-1}
             \bigl[(WJ)^T W \hat d + \mu \alpha_s m_{\mathrm{ref}}\bigr],
    \qquad \hat d = d - f(m_k) + J m_k,

where :math:`R = \partial^T \partial + \alpha_s I` combines first-difference
roughness :math:`\partial` with an optional smallness term. While the target
:math:`\chi^2 = N_d` cannot be reached, :math:`\mu` minimizes the nonlinear
misfit and only steps that reduce it are accepted; once the target can be
reached, :math:`\mu` is the largest value with :math:`\chi^2 = N_d`.

The depth of investigation follows Oldenburg and Li (1999): two further
inversions with reference models of 0.1 and 10 times the geometric mean of the
apparent resistivities give, layer by layer, the index

.. math::

    R^{\mathrm{DOI}}_j = \frac{m^{(1)}_j - m^{(2)}_j}
                             {m^{(1)}_{\mathrm{ref}} - m^{(2)}_{\mathrm{ref}}},

which is near 0 where the data control the model and near 1 where the
reference does. The index is rescaled to 1 at the bottom layer. Near the
transition the two reference models can overshoot in opposite directions,
making the index negative there, so its magnitude is used. The depth of
investigation is the top of the unresolved zone that reaches the bottom of
the model: going up from the bottom layer, through layers whose rescaled index
is at least 0.1, bridging resolved gaps narrower than a factor of 2 in depth.
Bridging ignores the dips below 0.1 that the overshoot produces inside the
unresolved zone, while a genuinely resolved deeper zone (for example a
conductive layer below a resistive one whose resistivity and thickness are
equivalent) is not swallowed by an unresolved band above it. The reference
inversions aim at the same misfit as the main inversion.
"""

from __future__ import annotations

import operator
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

from ertkit._linalg import matmul
from ertkit._warnings import ConvergenceWarning, DataWarning
from ertkit.results import InversionResult
from ertkit.ves.data import Sounding
from ertkit.ves.forward import _rhoa
from ertkit.ves.invert import _POOR_FIT_P_VALUE, _segments, _static_shift_hint
from ertkit.ves.model import LayeredModel

__all__ = ["DOI_THRESHOLD", "SmoothFit", "invert_smooth"]

_STEP = 1e-30
DOI_THRESHOLD = 0.1
"""Rescaled depth-of-investigation index at which a layer counts as unresolved
(Oldenburg and Li 1999)."""
_SMOOTHEST_FITS = "the smoothest model fits better than the target misfit"
_TARGET_REACHED = "target misfit reached and the model is stable"
_TARGET_UNREACHABLE = "the target misfit cannot be reached; the misfit was minimized"
_TARGET_RELAXED = (
    "chi2 = N_d cannot be reached; the model fits 1.1 times the smallest misfit"
)
_REFIT_FACTOR = 1.1
"""When chi2 = N_d cannot be reached, the misfit targeted instead, as a multiple
of the smallest misfit found (a smooth model rather than the oscillating
misfit-minimizing one)."""
_ALPHA_S_DOI = 0.1
"""Smallness weight, relative to roughness, in the two reference inversions,
for 30 layers; it is scaled with the layer density (see ``_alpha_s``)."""
_REFERENCE_FACTORS = (0.1, 10.0)
"""Reference models, as multiples of the geometric mean apparent resistivity."""
_DOI_GAP_FACTOR = 2.0
"""Resolved gaps narrower than this factor in depth, inside the unresolved zone
that reaches the bottom of the model, are bridged when the depth of
investigation is placed."""


@dataclass(frozen=True, eq=False, kw_only=True, slots=True)
class SmoothFit(InversionResult):
    r"""Result of a smooth (Occam) inversion.

    Attributes
    ----------
    model : LayeredModel
        The smooth model.
    mu : float
        Final regularization weight :math:`\mu`.
    sensitivity : ndarray
        Error-weighted sensitivity of the data to each layer,
        :math:`\lVert J_{:,j} / s \rVert`, with :math:`J` the Jacobian of
        :math:`\ln \rho_a` with respect to :math:`\ln \rho_j`. It is per
        layer, not per meter; since the layers thicken with depth, it is
        roughly a sensitivity per unit of logarithmic depth.
    doi_index : ndarray
        Magnitude of the rescaled Oldenburg and Li index of each layer: near 0
        where the data control the model, near 1 where they do not. ``nan`` if
        the depth of investigation was not computed or could not be
        determined.
    doi : float
        Depth of investigation in meters: the top of the unresolved zone that
        reaches the bottom of the model (rescaled index at least
        :data:`DOI_THRESHOLD`, bridging resolved gaps narrower than a factor
        of 2 in depth). Below it the model reflects the regularization. Some
        shallower layers can also be unresolved (for example inside a thin
        resistive layer whose thickness and resistivity are equivalent);
        plots shade them too. ``0.0`` if no layer is resolved, ``nan`` if not
        computed or not determined.

    See Also
    --------
    ertkit.InversionResult : The attributes common to all results.
    """

    model: LayeredModel
    mu: float
    sensitivity: NDArray[np.float64]
    doi_index: NDArray[np.float64]
    doi: float

    def _details(self) -> list[str]:
        doi = "not computed" if np.isnan(self.doi) else f"{self.doi:.3g} m"
        return [
            (
                f"{self.model.n_layers} layers, the deepest interface at "
                f"{self.model.depths[-1]:.3g} m; mu = {self.mu:.3g}"
            ),
            f"depth of investigation: {doi}",
        ]


def _layer_thicknesses(
    data: Sounding, n_layers: int, depth_range: tuple[float, float] | None
) -> NDArray[np.float64]:
    if depth_range is None:
        # From well above the shallowest sensitivity to well below the deepest.
        depth_range = (
            0.3 * float(data.survey.ab2.min()),
            3.0 * float(data.survey.ab2.max()),
        )
    top, bottom = depth_range
    if not (0 < top < bottom and np.isfinite(bottom)):
        raise ValueError("depth_range must satisfy 0 < top < bottom")
    interfaces = np.geomspace(top, bottom, n_layers - 1)
    return np.diff(np.concatenate([[0.0], interfaces]))


class _Occam:
    """One smooth inversion problem: data, layers and regularization."""

    def __init__(
        self,
        data: Sounding,
        thickness: NDArray[np.float64],
        alpha_s: float,
        m_ref: float,
    ) -> None:
        self.data = data
        self.h = thickness
        self.n = thickness.size + 1
        self.d = data.log_data
        self.w = 1.0 / data.log_std
        diff = np.diff(np.eye(self.n), axis=0)
        self.alpha_s = alpha_s
        self.reg = matmul(diff.T, diff) + alpha_s * np.eye(self.n)
        self.reg_rhs = alpha_s * m_ref * np.ones(self.n)
        # Models outside this range cannot be meaningful for these data, and
        # far outside it the forward model overflows.
        self.m_lo = float(np.log(data.rhoa.min() / 1e3))
        self.m_hi = float(np.log(data.rhoa.max() * 1e3))

    def predict(
        self, m: NDArray[np.float64] | NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        rhoa = _rhoa(np.exp(m), self.h, self.data.survey.ab2, self.data.survey.mn2)
        return np.asarray(np.log(rhoa.astype(complex)))

    def chi2(self, m: NDArray[np.float64]) -> float:
        """Misfit of model m; ``inf`` for models outside the meaningful range."""
        if not np.all(np.isfinite(m)) or m.min() < self.m_lo or m.max() > self.m_hi:
            return float("inf")
        with np.errstate(all="ignore"):
            value = float(np.sum((self.w * (self.d - self.predict(m).real)) ** 2))
        return value if np.isfinite(value) else float("inf")

    def jacobian(self, m: NDArray[np.float64]) -> NDArray[np.float64]:
        jac = np.empty((self.d.size, self.n))
        for k in range(self.n):
            bumped = m.astype(complex)
            bumped[k] += 1j * _STEP
            jac[:, k] = self.predict(bumped).imag / _STEP
        return jac

    def model_for(
        self, mu: float, gram: NDArray[np.float64], rhs: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.asarray(
            np.linalg.solve(gram + mu * self.reg, rhs + mu * self.reg_rhs)
        )

    def step(
        self, m: NDArray[np.float64], target: float
    ) -> tuple[NDArray[np.float64], float, bool, bool]:
        """One Occam iteration.

        Returns the new model, mu, whether the target is reachable, and whether
        even the smoothest model tried fits better than the target.
        """
        jac = self.jacobian(m)
        wj = self.w[:, np.newaxis] * jac
        gram = matmul(wj.T, wj)
        d_hat = self.d - self.predict(m).real + matmul(jac, m)
        rhs = matmul(wj.T, self.w * d_hat)
        mu_ref = float(np.trace(gram) / np.trace(self.reg))

        def misfit(log_mu: float) -> float:
            return self.chi2(self.model_for(mu_ref * 10.0**log_mu, gram, rhs))

        grid = np.linspace(-6.0, 6.0, 49)
        values = np.array([misfit(x) for x in grid])
        reachable = bool(np.any(values <= target))
        smoothest = bool(values[-1] <= target)
        if not np.any(np.isfinite(values)):
            return m, float("nan"), False, False
        if not reachable:
            i = int(np.argmin(values))
            lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
            best = optimize.minimize_scalar(misfit, bounds=(lo, hi), method="bounded")
            log_mu = float(best.x) if best.fun <= values[i] else float(grid[i])
            if best.fun <= target:
                # The grid missed a narrow range of mu that reaches the target:
                # take the largest mu with chi2 = target between the refined
                # minimum and the next grid point above it.
                reachable = True
                above = grid[(grid > log_mu) & (values > target)]
                if above.size:
                    log_mu = float(
                        optimize.brentq(
                            lambda x: misfit(x) - target, log_mu, above[0], xtol=1e-6
                        )
                    )
        else:
            below = np.flatnonzero(values <= target)
            i = int(below.max())
            if i == grid.size - 1:
                log_mu = float(grid[-1])
            else:
                log_mu = float(
                    optimize.brentq(
                        lambda x: misfit(x) - target, grid[i], grid[i + 1], xtol=1e-6
                    )
                )
        mu = mu_ref * 10.0**log_mu
        return self.model_for(mu, gram, rhs), mu, reachable, smoothest

    def run(
        self, m0: NDArray[np.float64], max_iter: int, target: float | None = None
    ) -> tuple[NDArray[np.float64], float, bool, str, int]:
        target = float(self.d.size) if target is None else float(target)
        m, mu = m0, float("nan")
        current = self.chi2(m)
        for k in range(1, max_iter + 1):
            new, new_mu, reachable, smoothest = self.step(m, target)
            new_chi2 = self.chi2(new)
            if not reachable:
                # Accept only steps that reduce the misfit; halve otherwise.
                for _ in range(20):
                    if new_chi2 < current:
                        break
                    new = m + 0.5 * (new - m)
                    new_chi2 = self.chi2(new)
                decrease = (
                    (current - new_chi2) / current if np.isfinite(current) else 1.0
                )
                if not new_chi2 < current or decrease < 1e-3:
                    if new_chi2 < current:
                        m, mu, current = new, new_mu, new_chi2
                    return m, mu, True, _TARGET_UNREACHABLE, k
            change = float(np.linalg.norm(new - m) / np.sqrt(self.n))
            m, mu, current = new, new_mu, new_chi2
            if change < 1e-3 and smoothest:
                return m, mu, True, _SMOOTHEST_FITS, k
            if reachable and abs(current - target) <= 0.01 * target and change < 1e-3:
                return m, mu, True, _TARGET_REACHED, k
        return m, mu, False, f"stopped after {max_iter} iterations", max_iter


def _alpha_s(n_layers: int) -> float:
    """Smallness weight for the reference inversions.

    Scaled with the square of the layer density, so that the reference models
    act over the same depth range whatever the number of layers.
    """
    return _ALPHA_S_DOI * (28.0 / max(n_layers - 2, 1)) ** 2


def _doi(
    data: Sounding,
    thickness: NDArray[np.float64],
    max_iter: int,
    target: float,
) -> tuple[NDArray[np.float64], float]:
    """Oldenburg and Li (1999) index and depth of investigation.

    The two reference inversions aim at the same misfit as the main inversion.
    """
    mean = float(np.exp(np.mean(data.log_data)))
    refs = tuple(np.log(factor * mean) for factor in _REFERENCE_FACTORS)
    models = []
    for ref in refs:
        problem = _Occam(data, thickness, _alpha_s(thickness.size + 1), ref)
        m, _, converged, reason, _ = problem.run(
            np.full(problem.n, ref), max_iter, target=target
        )
        if not converged:
            warnings.warn(
                f"a reference inversion for the depth of investigation did not "
                f"converge ({reason}); the depth of investigation may be unreliable",
                ConvergenceWarning,
                stacklevel=3,
            )
        models.append(m)
    index = (models[0] - models[1]) / (refs[0] - refs[1])
    bottom = float(index[-1])
    if bottom <= 0.2:
        warnings.warn(
            "the depth of investigation could not be determined: the reference "
            "models still agree at the bottom of the model; increase depth_range",
            DataWarning,
            stacklevel=3,
        )
        return np.full(index.size, np.nan), float("nan")
    scaled = np.abs(index) / bottom
    return scaled, _doi_depth(scaled, np.concatenate([[0.0], np.cumsum(thickness)]))


def _doi_depth(scaled: NDArray[np.float64], tops: NDArray[np.float64]) -> float:
    """Top of the unresolved zone that reaches the bottom of the model.

    Starting from the bottom layer (index 1 by construction), the zone extends
    upward through layers whose index reaches the threshold, bridging resolved
    gaps narrower than ``_DOI_GAP_FACTOR`` in depth. A wider resolved gap ends
    it; unresolved layers above that gap do not set the depth of investigation
    (plots shade them separately).
    """
    unresolved = scaled >= DOI_THRESHOLD
    if np.all(unresolved):
        return 0.0
    j = unresolved.size - 1
    while j > 0:
        if unresolved[j - 1]:
            j -= 1
            continue
        k = j - 1  # find the top of the resolved gap above layer j
        while k >= 0 and not unresolved[k]:
            k -= 1
        if k >= 0 and tops[j] < _DOI_GAP_FACTOR * max(tops[k + 1], tops[1]):
            j = k  # a narrow gap with unresolved layers above it: bridge it
            continue
        break
    return float(tops[j])


def invert_smooth(
    data: Sounding,
    *,
    n_layers: int = 30,
    depth_range: tuple[float, float] | None = None,
    depth_of_investigation: bool = True,
    max_iter: int = 30,
) -> SmoothFit:
    r"""Smooth (Occam) inversion with sensitivity and depth of investigation.

    Parameters
    ----------
    data : Sounding
        The data and their error model.
    n_layers : int, optional
        Number of layers, including the bottom half-space. Default 30.
    depth_range : (float, float), optional
        Depths in meters of the first and last layer interfaces, which are
        spaced logarithmically. Default: 0.3 times the smallest and 3 times
        the largest AB/2.
    depth_of_investigation : bool, optional
        Also compute the Oldenburg and Li depth of investigation (two further
        inversions). Default True.
    max_iter : int, optional
        Maximum number of Occam iterations. Default 30.

    Returns
    -------
    SmoothFit

    Raises
    ------
    TypeError
        If ``data`` is not a :class:`Sounding`.
    ValueError
        If ``n_layers`` is less than 2, ``max_iter`` is less than 1, or
        ``depth_range`` is invalid.

    Warns
    -----
    ConvergenceWarning
        If the iterations stop before the model is stable, or a reference
        inversion for the depth of investigation does not converge.
    DataWarning
        If even the smallest misfit is improbably large (goodness-of-fit
        p-value below 0.01: the errors may be underestimated), if even the
        smoothest model fits better than the target (the errors may be
        overestimated), or if the depth of investigation cannot be determined
        (extend ``depth_range``).

    Notes
    -----
    The method is Occam's inversion (Constable et al. 1987). If no model
    reaches :math:`\chi^2 = N_d` (which happens for a few percent of soundings
    even with correct errors), the inversion fits 1.1 times the smallest
    misfit instead, recorded in ``chi2_target`` and ``stop_reason``, and warns
    only if that misfit is improbably large.

    The depth of investigation follows Oldenburg and Li (1999), with reference
    models of 0.1 and 10 times the geometric mean apparent resistivity (a
    factor of 100 apart) that aim at the same misfit. The module
    documentation gives the details.

    Examples
    --------
    >>> import numpy as np
    >>> from ertkit.ves import LayeredModel, Sounding, Survey, invert_smooth
    >>> truth = LayeredModel(resistivity=[150.0, 30.0, 800.0], thickness=[2.0, 15.0])
    >>> survey = Survey.schlumberger(ab2=np.geomspace(2.5, 200.0, 22), mn2=0.5)
    >>> smooth = invert_smooth(Sounding.synthetic(truth, survey, rng=5))
    >>> smooth.model.n_layers, smooth.converged
    (30, True)
    """
    data_obj: object = data
    if not isinstance(data_obj, Sounding):
        raise TypeError(f"data must be a Sounding, got {type(data_obj).__name__}")
    n_layers = operator.index(n_layers)
    max_iter = operator.index(max_iter)
    if n_layers < 2:
        raise ValueError("a smooth model needs at least 2 layers")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    thickness = _layer_thicknesses(data, n_layers, depth_range)
    start_value = float(np.mean(data.log_data))
    problem = _Occam(data, thickness, alpha_s=0.0, m_ref=start_value)
    start = np.full(n_layers, start_value)
    target = float(data.n_data)
    m, mu, converged, reason, n_iter = problem.run(start, max_iter)
    if converged and reason == _TARGET_UNREACHABLE:
        # The misfit-minimizing model oscillates; fit a little less closely.
        chi2_min = problem.chi2(m)
        target = _REFIT_FACTOR * chi2_min
        m, mu, converged, refit_reason, refit_iter = problem.run(
            start, max_iter, target=target
        )
        n_iter += refit_iter
        reason = _TARGET_RELAXED if converged else refit_reason
        if stats.chi2.sf(chi2_min, data.n_data) < _POOR_FIT_P_VALUE:
            n_segments = int(_segments(data.survey.mn2).max()) + 1
            warnings.warn(
                f"poor fit: the smallest misfit is chi2 = {chi2_min:.1f} for "
                f"{data.n_data} data; the data errors may be underestimated, or "
                f"the earth is not layered{_static_shift_hint(n_segments)}",
                DataWarning,
                stacklevel=2,
            )
    if not converged:
        warnings.warn(f"Occam inversion: {reason}", ConvergenceWarning, stacklevel=2)
    elif reason == _SMOOTHEST_FITS:
        warnings.warn(
            f"{reason}: the data errors may be overestimated", DataWarning, stacklevel=2
        )

    jac = problem.jacobian(m)
    sensitivity = np.linalg.norm(problem.w[:, np.newaxis] * jac, axis=0)
    if depth_of_investigation:
        doi_index, doi = _doi(data, thickness, max_iter, target)
    else:
        doi_index, doi = np.full(n_layers, np.nan), float("nan")

    predicted = problem.predict(m).real
    return SmoothFit(
        predicted=np.exp(predicted),
        residuals=(data.log_data - predicted) / data.log_std,
        chi2=problem.chi2(m),
        chi2_target=target,
        n_data=data.n_data,
        converged=converged,
        stop_reason=reason,
        n_iterations=n_iter,
        model=LayeredModel(resistivity=np.exp(m), thickness=thickness),
        mu=float(mu),
        sensitivity=sensitivity,
        doi_index=doi_index,
        doi=doi,
    )
