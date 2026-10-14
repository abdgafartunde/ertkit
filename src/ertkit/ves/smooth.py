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
misfit; once it can, :math:`\mu` is the largest value with
:math:`\chi^2 = N_d`.

The depth of investigation follows Oldenburg and Li (1999): two further
inversions with reference models of 0.1 and 10 times the mean apparent
resistivity (Marescot et al. 2003) give, layer by layer,
:math:`R_j = (m^{(1)}_j - m^{(2)}_j)/(m^{(1)}_{\mathrm{ref}} - m^{(2)}_{\mathrm{ref}})`,
which is near 0 where the data control the model and near 1 where the
reference does. :math:`R` is rescaled to 1 at the bottom layer. Near the
transition the two reference models can overshoot in opposite directions,
making :math:`R` negative there, so the magnitude :math:`|R|` is used: the
depth of investigation is the depth below which the rescaled :math:`|R|`
stays above 0.1.
"""

from __future__ import annotations

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

__all__ = ["SmoothFit", "invert_smooth"]

_STEP = 1e-30
DOI_THRESHOLD = 0.1
"""Rescaled index above which a layer counts as unresolved (Oldenburg and Li 1999)."""
SMOOTHEST_FITS = "the smoothest model fits better than the target misfit"
_ALPHA_S_DOI = 0.1
"""Smallness weight, relative to roughness, in the two reference inversions."""


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
        :math:`S_j = \lVert J_{:,j} / s \rVert` with :math:`J` the Jacobian of
        :math:`\ln \rho_a` with respect to :math:`\ln \rho_j`.
    doi_index : ndarray
        Magnitude of the rescaled Oldenburg and Li index of each layer: near 0
        where the data control the model, near 1 where they do not. ``nan`` if the depth of
        investigation was not computed or could not be determined.
    doi : float
        Depth of investigation in meters: below it, the model reflects the
        regularization rather than the data. ``nan`` if not computed or not
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

    def predict(
        self, m: NDArray[np.float64] | NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        rhoa = _rhoa(np.exp(m), self.h, self.data.survey.ab2, self.data.survey.mn2)
        return np.asarray(np.log(rhoa.astype(complex)))

    def chi2(self, m: NDArray[np.float64]) -> float:
        return float(np.sum((self.w * (self.d - self.predict(m).real)) ** 2))

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
        if not reachable:
            i = int(np.argmin(values))
            lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
            best = optimize.minimize_scalar(misfit, bounds=(lo, hi), method="bounded")
            log_mu = float(best.x)
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
        m, mu = m0, np.nan
        for k in range(1, max_iter + 1):
            new, mu, reachable, smoothest = self.step(m, target)
            change = float(np.linalg.norm(new - m) / np.sqrt(self.n))
            m = new
            chi2 = self.chi2(m)
            if change < 1e-3 and smoothest:
                return m, mu, True, SMOOTHEST_FITS, k
            if reachable and abs(chi2 - target) <= 0.01 * target and change < 1e-3:
                return m, mu, True, "target misfit reached and the model is stable", k
        return m, mu, False, f"stopped after {max_iter} iterations", max_iter


def _doi(
    data: Sounding,
    thickness: NDArray[np.float64],
    max_iter: int,
) -> tuple[NDArray[np.float64], float]:
    """Oldenburg and Li (1999) index and depth of investigation."""
    mean = float(np.exp(np.mean(data.log_data)))
    refs = (np.log(0.1 * mean), np.log(10.0 * mean))
    models = []
    for ref in refs:
        problem = _Occam(data, thickness, _ALPHA_S_DOI, ref)
        m, *_ = problem.run(np.full(problem.n, ref), max_iter)
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
    # Depth below which every layer is unresolved; layer j spans depth tops[j] to tops[j+1].
    tops = np.concatenate([[0.0], np.cumsum(thickness)])
    resolved = np.flatnonzero(scaled < DOI_THRESHOLD)
    if resolved.size == 0:
        return scaled, 0.0
    # The bottom layer has scaled index 1 by construction, so it is never
    # resolved and tops[deepest + 1] always exists.
    deepest = int(resolved.max())
    return scaled, float(tops[deepest + 1])


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
        If ``n_layers`` is less than 2 or ``depth_range`` is invalid.

    Warns
    -----
    ConvergenceWarning
        If the target misfit is not reached with a stable model.
    DataWarning
        If the depth of investigation cannot be determined, or if even the
        smoothest model fits better than the target (the data errors may be
        overestimated).
    """
    if n_layers < 2:
        raise ValueError("a smooth model needs at least 2 layers")
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
