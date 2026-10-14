r"""Inversion of VES data for layered-earth models.

The few-layer inversion estimates :math:`m = (\ln \rho_i, \ln h_i)` by
minimizing the misfit of the log data,

.. math::

    \chi^2 = \sum_{i=1}^{N_d} \left( \frac{d_i - f_i(m)}{s_i} \right)^2,
    \qquad d_i = \ln \rho_{a,i},

with a Levenberg-Marquardt-type trust-region method
(:func:`scipy.optimize.least_squares`) from several starting models. The
Jacobian is computed by complex-step differentiation, which is exact to
round-off.
"""

from __future__ import annotations

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
from ertkit.ves.model import LayeredModel

__all__ = ["LayerFit", "invert_few_layer"]

_STEP = 1e-30  # complex step; any value far below sqrt(machine epsilon) works


@dataclass(frozen=True, eq=False, kw_only=True, slots=True)
class LayerFit(InversionResult):
    r"""Result of a few-layer inversion.

    Attributes
    ----------
    model : LayeredModel
        The best-fitting model.
    parameter_names : tuple of str
        Names of the estimated parameters, in order: ``rho_1``, ...,
        ``rho_N``, ``h_1``, ..., ``h_{N-1}``, then ``shift_2``, ... for the
        static-shift factors of the second and later MN segments.
    std_log : ndarray
        Linearized standard deviation of the natural logarithm of each
        parameter.
    correlation : ndarray
        Linearized correlation matrix of the log parameters.
    depths : ndarray
        Depths to the bottoms of the upper :math:`N - 1` layers, in meters.
    depth_std : ndarray
        Linearized standard deviations of ``depths``, in meters.
    static_shift : ndarray
        Multiplicative static-shift factor of each MN segment (the first is 1
        by definition). All ones when shifts were not estimated.
    p_value : float
        Probability of a misfit at least this large if the model and the error
        model were right (:math:`\chi^2` distribution with :math:`N_d - p`
        degrees of freedom).
    n_starts : int
        Number of starting models tried.
    at_bound : tuple of str
        Parameters that ended at their search bounds, which the data do not
        resolve (see :func:`invert_few_layer`).
    """

    model: LayeredModel
    parameter_names: tuple[str, ...]
    std_log: NDArray[np.float64]
    correlation: NDArray[np.float64]
    depths: NDArray[np.float64]
    depth_std: NDArray[np.float64]
    static_shift: NDArray[np.float64]
    p_value: float
    n_starts: int
    at_bound: tuple[str, ...]

    @property
    def factor(self) -> NDArray[np.float64]:
        r"""Multiplicative uncertainty :math:`e^{s}` of each parameter.

        A factor of 1.2 means the parameter is known within a factor of 1.2
        (one standard deviation, linearized). Unresolved parameters give
        ``inf``.
        """
        with np.errstate(over="ignore"):
            return np.exp(self.std_log)

    @property
    def covariance(self) -> NDArray[np.float64]:
        """Linearized covariance matrix of the log parameters."""
        return self.correlation * np.outer(self.std_log, self.std_log)


def _segments(mn2: NDArray[np.float64]) -> NDArray[np.intp]:
    """Index of the MN segment of each reading (a new segment where MN/2 changes)."""
    changes = np.concatenate([[False], mn2[1:] != mn2[:-1]])
    return np.cumsum(changes).astype(np.intp)


class _Problem:
    """Log-space forward model and Jacobian of a few-layer sounding."""

    def __init__(self, data: Sounding, n_layers: int, static_shift: bool) -> None:
        self.n_layers = n_layers
        self.ab2 = data.survey.ab2
        self.mn2 = data.survey.mn2
        self.segment = _segments(self.mn2)
        self.n_shifts = int(self.segment.max()) if static_shift else 0
        self.n_params = 2 * n_layers - 1 + self.n_shifts
        self.d = data.log_data
        self.s = data.log_std

    def predict(
        self, m: NDArray[np.complex128] | NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        n = self.n_layers
        p = np.exp(m[: 2 * n - 1])
        f = np.log(_rhoa(p[:n], p[n:], self.ab2, self.mn2).astype(complex))
        if self.n_shifts:
            shifts = np.concatenate([[0.0], m[2 * n - 1 :]])
            f = f + shifts[self.segment]
        return np.asarray(f)

    def residual(self, m: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray((self.predict(m).real - self.d) / self.s)

    def jacobian(self, m: NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian of the log data with respect to m, by complex steps."""
        jac = np.empty((self.d.size, m.size))
        for k in range(m.size):
            bumped = m.astype(complex)
            bumped[k] += 1j * _STEP
            jac[:, k] = self.predict(bumped).imag / _STEP
        return jac

    def bounds(self, data: Sounding) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Search bounds on the log parameters.

        Resistivities within a factor of 1000 of the observed apparent
        resistivities, thicknesses between 1/1000 of the smallest and 10 times
        the largest AB/2, and static shifts within a factor of 2: the data
        cannot resolve values beyond these.
        """
        n = self.n_layers
        rho_lo, rho_hi = np.log(data.rhoa.min() / 1e3), np.log(data.rhoa.max() * 1e3)
        h_lo, h_hi = np.log(self.ab2.min() / 1e3), np.log(self.ab2.max() * 10.0)
        lower = np.concatenate(
            [
                np.full(n, rho_lo),
                np.full(n - 1, h_lo),
                np.full(self.n_shifts, -np.log(2.0)),
            ]
        )
        upper = np.concatenate(
            [
                np.full(n, rho_hi),
                np.full(n - 1, h_hi),
                np.full(self.n_shifts, np.log(2.0)),
            ]
        )
        return lower, upper

    def names(self) -> tuple[str, ...]:
        n = self.n_layers
        return (
            tuple(f"rho_{i + 1}" for i in range(n))
            + tuple(f"h_{i + 1}" for i in range(n - 1))
            + tuple(f"shift_{j + 2}" for j in range(self.n_shifts))
        )


def _random_start(
    data: Sounding, n_layers: int, n_shifts: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Log-uniform random model spanning the data's resistivities and depths."""
    lo, hi = np.log(data.rhoa.min() / 3), np.log(data.rhoa.max() * 3)
    log_rho = rng.uniform(lo, hi, n_layers)
    # Depths of investigation are roughly AB/2 / 3 for Schlumberger soundings.
    zmin, zmax = np.log(data.survey.ab2.min() / 3), np.log(data.survey.ab2.max() / 3)
    depths = np.exp(np.sort(rng.uniform(zmin, zmax, n_layers - 1)))
    log_h = np.log(np.diff(np.concatenate([[0.0], depths])))
    return np.concatenate([log_rho, log_h, np.zeros(n_shifts)])


def invert_few_layer(
    data: Sounding,
    n_layers: int,
    *,
    start: LayeredModel | None = None,
    n_starts: int = 10,
    static_shift: bool = False,
    rng: int | np.random.Generator | None = None,
) -> LayerFit:
    r"""Fit a layered model with a few layers to a sounding.

    Parameters
    ----------
    data : Sounding
        The data and their error model.
    n_layers : int
        Number of layers :math:`N`, including the bottom half-space.
    start : LayeredModel, optional
        Starting model. If given, only this start is used; otherwise
        ``n_starts`` random starting models are tried and the best fit kept.
    n_starts : int, optional
        Number of random starting models. Default 10.
    static_shift : bool, optional
        Also estimate a multiplicative static-shift factor for every MN
        segment after the first (segments start where :math:`MN/2` changes).
        Default False.
    rng : int, numpy.random.Generator or None, optional
        Seed or generator for the starting models.

    Returns
    -------
    LayerFit

    Raises
    ------
    ValueError
        If there are not more data than parameters, or ``start`` has the wrong
        number of layers.

    Warns
    -----
    ConvergenceWarning
        If no start converged.
    DataWarning
        If the fit is poor (goodness-of-fit p-value below 0.01): the error
        model underestimates the errors, or the model has too few layers.

    Notes
    -----
    The linearized covariance of the log parameters is
    :math:`(J^T W^2 J)^{-1}` with :math:`W = \mathrm{diag}(1/s_i)`. It
    describes the uncertainty well only when the parameters are well
    resolved; for equivalent layers, use the equivalence analysis.
    """
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    problem = _Problem(data, n_layers, static_shift)
    if data.n_data <= problem.n_params:
        raise ValueError(
            f"{data.n_data} data cannot determine {problem.n_params} parameters"
        )
    generator = np.random.default_rng(rng)
    if start is not None:
        if start.n_layers != n_layers:
            raise ValueError(f"start has {start.n_layers} layers, expected {n_layers}")
        starts = [
            np.concatenate(
                [
                    np.log(start.resistivity),
                    np.log(start.thickness),
                    np.zeros(problem.n_shifts),
                ]
            )
        ]
    else:
        starts = [
            _random_start(data, n_layers, problem.n_shifts, generator)
            for _ in range(n_starts)
        ]

    lower, upper = problem.bounds(data)
    margin = 1e-6 * (upper - lower)

    # scipy passes extra arguments to these callables, so they accept them.
    def residual(
        m: NDArray[np.float64], *_: object, **__: object
    ) -> NDArray[np.float64]:
        return problem.residual(m)

    def jacobian(
        m: NDArray[np.float64], *_: object, **__: object
    ) -> NDArray[np.float64]:
        return problem.jacobian(m) / problem.s[:, np.newaxis]

    best: optimize.OptimizeResult | None = None
    for x0 in starts:
        fit = optimize.least_squares(
            residual,
            np.clip(x0, lower + margin, upper - margin),
            jac=jacobian,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=200 * problem.n_params,
        )
        if best is None or (fit.success, -fit.cost) > (best.success, -best.cost):
            best = fit
    assert best is not None
    if not best.success:
        warnings.warn(
            f"no starting model converged: {best.message}",
            ConvergenceWarning,
            stacklevel=2,
        )

    m = best.x
    n = n_layers
    tol = 1e-4 * (upper - lower)
    at_bound = tuple(
        name
        for name, value, lo, hi, t in zip(
            problem.names(), m, lower, upper, tol, strict=True
        )
        if value <= lo + t or value >= hi - t
    )
    if at_bound:
        warnings.warn(
            f"parameters at their search bounds, not resolved by the data: "
            f"{', '.join(at_bound)}; consider fewer layers",
            DataWarning,
            stacklevel=2,
        )
    chi2 = float(2.0 * best.cost)
    dof = data.n_data - problem.n_params
    p_value = float(stats.chi2.sf(chi2, dof))
    if p_value < 0.01:
        warnings.warn(
            f"poor fit: chi2 = {chi2:.1f} for {dof} degrees of freedom (p = {p_value:.2g}); "
            "the errors may be underestimated, or more layers may be needed",
            DataWarning,
            stacklevel=2,
        )

    weighted = problem.jacobian(m) / problem.s[:, np.newaxis]
    covariance = np.linalg.pinv(matmul(weighted.T, weighted))
    std_log = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.outer(std_log, std_log)
    resistivity, thickness = np.exp(m[:n]), np.exp(m[n : 2 * n - 1])
    # Depth z_i = h_1 + ... + h_i; its gradient with respect to ln h_j is h_j (j <= i).
    grad = np.tril(np.ones((n - 1, n - 1))) * thickness[np.newaxis, :]
    cov_h = covariance[n : 2 * n - 1, n : 2 * n - 1]
    depth_std = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", grad, cov_h, grad), 0.0, None))
    shifts = np.exp(np.concatenate([[0.0], m[2 * n - 1 :]]))
    if not problem.n_shifts:
        shifts = np.ones(int(problem.segment.max()) + 1)

    return LayerFit(
        predicted=np.exp(problem.predict(m).real),
        chi2=chi2,
        chi2_target=float(dof),
        n_data=data.n_data,
        converged=bool(best.success),
        stop_reason=str(best.message),
        n_iterations=int(best.nfev),
        model=LayeredModel(resistivity=resistivity, thickness=thickness),
        parameter_names=problem.names(),
        std_log=std_log,
        correlation=correlation,
        depths=np.cumsum(thickness),
        depth_std=depth_std,
        static_shift=shifts,
        p_value=p_value,
        n_starts=len(starts),
        at_bound=at_bound,
    )
