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
apparent resistivities (Marescot et al. 2003) give, layer by layer, the index

.. math::

    R^{\mathrm{DOI}}_j = \frac{m^{(1)}_j - m^{(2)}_j}
                             {m^{(1)}_{\mathrm{ref}} - m^{(2)}_{\mathrm{ref}}},

which is near 0 where the data control the model and near 1 where the
reference does. The index is rescaled to 1 at the bottom layer. Near the
transition the two reference models can overshoot in opposite directions,
making the index negative there, so its magnitude is used. The depth of
investigation is the shallowest depth from which the rescaled index stays at
or above 0.1 over at least a factor of 1.5 in depth. Requiring this
persistence ignores an isolated layer whose index fluctuates just above the
threshold, while still placing the depth where resolution is lost rather than
where the index finally stays high to the bottom of the model.
"""

from __future__ import annotations

import operator
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from ertkit._linalg import matmul
from ertkit._warnings import ConvergenceWarning, DataWarning
from ertkit.results import InversionResult
from ertkit.ves.data import Sounding
from ertkit.ves.forward import _rhoa
from ertkit.ves.model import LayeredModel

__all__ = ["DOI_THRESHOLD", "SmoothFit", "invert_smooth"]

_STEP = 1e-30
DOI_THRESHOLD = 0.1
"""Rescaled depth-of-investigation index at which a layer counts as unresolved
(Oldenburg and Li 1999)."""
SMOOTHEST_FITS = "the smoothest model fits better than the target misfit"
TARGET_REACHED = "target misfit reached and the model is stable"
TARGET_UNREACHABLE = "the target misfit cannot be reached; the misfit was minimized"
_ALPHA_S_DOI = 0.1
"""Smallness weight, relative to roughness, in the two reference inversions."""
_REFERENCE_FACTORS = (0.1, 10.0)
DOI_PERSISTENCE = 1.5
"""Depth factor over which the index must stay above the threshold."""
"""Reference models, as multiples of the geometric mean apparent resistivity."""


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
        Depth of investigation in meters: the shallowest depth from which the
        rescaled index stays at or above :data:`DOI_THRESHOLD` over at least a
        factor of 1.5 in depth. Below it the model reflects the
        regularization. An isolated shallower layer can also exceed the
        threshold; plots shade such layers too. ``nan`` if not computed or not
        determined.
    """

    model: LayeredModel
    mu: float
    sensitivity: NDArray[np.float64]
    doi_index: NDArray[np.float64]
    doi: float


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
        self, m0: NDArray[np.float64], max_iter: int
    ) -> tuple[NDArray[np.float64], float, bool, str, int]:
        target = float(self.d.size)
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
                    return m, mu, True, TARGET_UNREACHABLE, k
            change = float(np.linalg.norm(new - m) / np.sqrt(self.n))
            m, mu, current = new, new_mu, new_chi2
            if change < 1e-3 and smoothest:
                return m, mu, True, SMOOTHEST_FITS, k
            if reachable and abs(current - target) <= 0.01 * target and change < 1e-3:
                return m, mu, True, TARGET_REACHED, k
        return m, mu, False, f"stopped after {max_iter} iterations", max_iter


def _doi(
    data: Sounding,
    thickness: NDArray[np.float64],
    max_iter: int,
) -> tuple[NDArray[np.float64], float]:
    """Oldenburg and Li (1999) index and depth of investigation."""
    mean = float(np.exp(np.mean(data.log_data)))
    refs = tuple(np.log(factor * mean) for factor in _REFERENCE_FACTORS)
    models = []
    for ref in refs:
        problem = _Occam(data, thickness, _ALPHA_S_DOI, ref)
        m, _, converged, reason, _ = problem.run(np.full(problem.n, ref), max_iter)
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
    # Layer j spans depths tops[j] to tops[j + 1]. The first layer from which
    # the index stays above the threshold over a factor DOI_PERSISTENCE in
    # depth; the bottom layer (index 1) always qualifies.
    tops = np.concatenate([[0.0], np.cumsum(thickness)])
    for j in np.flatnonzero(scaled >= DOI_THRESHOLD):
        reach = DOI_PERSISTENCE * max(tops[j], tops[min(1, tops.size - 1)])
        window = (tops >= tops[j]) & (tops <= reach)
        if np.all(scaled[window] >= DOI_THRESHOLD):
            return scaled, float(tops[j])
    return scaled, float(tops[-1])  # pragma: no cover - the bottom layer qualifies


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
    ValueError
        If ``n_layers`` is less than 2, ``max_iter`` is less than 1, or
        ``depth_range`` is invalid.

    Warns
    -----
    ConvergenceWarning
        If the iterations stop before the model is stable, or a reference
        inversion for the depth of investigation does not converge.
    DataWarning
        If the target misfit cannot be reached (the errors may be
        underestimated), if even the smoothest model fits better than the
        target (the errors may be overestimated), or if the depth of
        investigation cannot be determined (extend ``depth_range``).

    Notes
    -----
    The method is Occam's inversion (Constable et al. 1987). The depth of
    investigation follows Oldenburg and Li (1999), with reference models of
    0.1 and 10 times the geometric mean apparent resistivity (Marescot et al.
    2003); the module documentation gives the details.

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
    n_layers = operator.index(n_layers)
    max_iter = operator.index(max_iter)
    if n_layers < 2:
        raise ValueError("a smooth model needs at least 2 layers")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    thickness = _layer_thicknesses(data, n_layers, depth_range)
    start_value = float(np.mean(data.log_data))
    problem = _Occam(data, thickness, alpha_s=0.0, m_ref=start_value)
    m, mu, converged, reason, n_iter = problem.run(
        np.full(n_layers, start_value), max_iter
    )
    if not converged:
        warnings.warn(f"Occam inversion: {reason}", ConvergenceWarning, stacklevel=2)
    elif reason == SMOOTHEST_FITS:
        warnings.warn(
            f"{reason}: the data errors may be overestimated", DataWarning, stacklevel=2
        )
    elif reason == TARGET_UNREACHABLE:
        warnings.warn(
            f"{reason} (chi2 = {problem.chi2(m):.1f} for {data.n_data} data): "
            "the data errors may be underestimated, or the earth is not layered",
            DataWarning,
            stacklevel=2,
        )

    jac = problem.jacobian(m)
    sensitivity = np.linalg.norm(problem.w[:, np.newaxis] * jac, axis=0)
    if depth_of_investigation:
        doi_index, doi = _doi(data, thickness, max_iter)
    else:
        doi_index, doi = np.full(n_layers, np.nan), float("nan")

    return SmoothFit(
        predicted=np.exp(problem.predict(m).real),
        chi2=problem.chi2(m),
        chi2_target=float(data.n_data),
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
