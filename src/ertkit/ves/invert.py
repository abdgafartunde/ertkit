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

import operator
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

from ertkit._warnings import ConvergenceWarning, DataWarning
from ertkit.results import InversionResult
from ertkit.ves.data import Sounding
from ertkit.ves.forward import _rhoa
from ertkit.ves.model import LayeredModel

__all__ = ["LayerFit", "invert_few_layer"]

_STEP = 1e-30  # complex step; any value far below sqrt(machine epsilon) works
POOR_FIT_P_VALUE = 0.01
"""Goodness-of-fit p-value below which a fit is reported as poor."""


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
        parameter; ``inf`` for parameters the data do not determine.
    covariance : ndarray
        Linearized covariance matrix of the log parameters; rows and columns
        of undetermined parameters are ``inf`` on the diagonal and ``nan``
        elsewhere.
    correlation : ndarray
        Linearized correlation matrix of the log parameters (``nan`` for
        undetermined parameters).
    depths : ndarray
        Depths to the bottoms of the upper :math:`N - 1` layers, in meters.
    depth_std : ndarray
        Linearized standard deviations of ``depths``, in meters (``inf`` when
        a thickness above is undetermined). When the standard deviation of a
        log thickness exceeds about 0.5, the linearization is poor and these
        values are only indicative.
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
        Parameters that ended at their search bounds (see the Notes of
        :func:`invert_few_layer`). The data do not resolve them, and their
        uncertainty factors are not meaningful.
    """

    model: LayeredModel
    parameter_names: tuple[str, ...]
    std_log: NDArray[np.float64]
    covariance: NDArray[np.float64]
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

        A factor of 1.2 means the parameter is known within a factor of 1.2,
        that is, multiplied or divided by 1.2 (one standard deviation,
        linearized). Parameters the data do not determine give ``inf``.
        """
        with np.errstate(over="ignore"):
            return np.exp(self.std_log)


def _segments(mn2: NDArray[np.float64]) -> NDArray[np.intp]:
    """Index of the MN segment of each reading.

    Readings with the same MN/2 form one segment, whatever their order in the
    file; segments are numbered in order of first appearance.
    """
    values, first, inverse = np.unique(mn2, return_index=True, return_inverse=True)
    rank = np.empty(values.size, dtype=np.intp)
    rank[np.argsort(first)] = np.arange(values.size)
    return rank[inverse].astype(np.intp)


def _linearized_covariance(
    weighted: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Covariance, standard deviations and correlations of the parameters.

    Computed from the singular value decomposition of :math:`WJ` itself
    rather than by inverting :math:`J^T W^2 J`, which would square the
    condition number and hide poorly determined directions. Directions with
    singular values below the numerical rank tolerance are undetermined, and
    every parameter loading on them gets an infinite standard deviation.
    """
    _, sv, vt = np.linalg.svd(weighted, full_matrices=False)
    tol = sv[0] * max(weighted.shape) * np.finfo(float).eps
    keep = sv > tol
    cov = (vt[keep].T / sv[keep] ** 2) @ vt[keep]
    unresolved = np.sum(vt[~keep] ** 2, axis=0) > 1e-8
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    std[unresolved] = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(std, std)
    for arr in (cov, corr):
        arr[unresolved, :] = np.nan
        arr[:, unresolved] = np.nan
    cov[unresolved, unresolved] = np.inf
    corr[~unresolved, ~unresolved] = 1.0
    return cov, std, corr


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
        ``n_starts`` random starting models are tried and the fit with the
        smallest misfit is kept.
    n_starts : int, optional
        Number of random starting models. Default 10.
    static_shift : bool, optional
        Also estimate a multiplicative static-shift factor for every MN
        segment after the first. Readings with the same MN/2 form a segment.
        Default False.
    rng : int, numpy.random.Generator or None, optional
        Seed or generator for the starting models.

    Returns
    -------
    LayerFit

    Raises
    ------
    ValueError
        If ``n_layers`` or ``n_starts`` is less than 1, if there are not more
        data than parameters, or if ``start`` has the wrong number of layers.

    Warns
    -----
    ConvergenceWarning
        If the best fit did not meet the convergence criterion.
    DataWarning
        If the fit is poor (goodness-of-fit p-value below 0.01: the errors may
        be underestimated, or the model needs more layers), or if parameters
        end at their search bounds (listed in ``LayerFit.at_bound``).

    Notes
    -----
    The parameters are searched within bounds beyond which the data cannot
    resolve them: resistivities within a factor of 1000 of the observed
    apparent resistivities, thicknesses between 1/1000 of the smallest and 10
    times the largest AB/2, and static shifts within a factor of 2.

    The linearized covariance of the log parameters is
    :math:`(J^T W^2 J)^{-1}` with :math:`W = \mathrm{diag}(1/s_i)`, computed
    from the singular values of :math:`WJ`. It describes the uncertainty well
    when the parameters are well resolved. Equivalent layers show up as large
    uncertainty factors, but a linearized uncertainty can still understate the
    range of acceptable models; the equivalence analysis planned for release
    0.2 samples that range directly.

    Examples
    --------
    >>> import numpy as np
    >>> from ertkit.ves import LayeredModel, Sounding, Survey, invert_few_layer
    >>> truth = LayeredModel(resistivity=[150.0, 30.0, 800.0], thickness=[2.0, 15.0])
    >>> survey = Survey.schlumberger(ab2=np.geomspace(2.5, 200.0, 22), mn2=0.5)
    >>> data = Sounding.synthetic(truth, survey, rng=5)
    >>> fit = invert_few_layer(data, n_layers=3, rng=0)
    >>> fit.parameter_names
    ('rho_1', 'rho_2', 'rho_3', 'h_1', 'h_2')
    """
    n_layers = operator.index(n_layers)
    n_starts = operator.index(n_starts)
    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")
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

    # Tolerances: 1e-8 in the log parameters is far below any uncertainty that
    # matters, and tighter values make starts on an equivalence ridge run out
    # of evaluations without improving the fit.
    fits = [
        optimize.least_squares(
            residual,
            np.clip(x0, lower + margin, upper - margin),
            jac=jacobian,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=1e-10,
            xtol=1e-8,
            gtol=1e-10,
            max_nfev=200 * problem.n_params,
        )
        for x0 in starts
    ]
    best = min(fits, key=lambda f: f.cost)
    converged = bool(
        best.success
        or any(f.success and f.cost <= best.cost * (1 + 1e-8) + 1e-12 for f in fits)
    )
    if not converged:
        warnings.warn(
            f"the best fit did not converge: {best.message}",
            ConvergenceWarning,
            stacklevel=2,
        )

    m = best.x
    n = n_layers
    names = problem.names()
    tol = 1e-4 * (upper - lower)
    at_bound = tuple(
        name
        for name, value, lo, hi, t in zip(names, m, lower, upper, tol, strict=True)
        if value <= lo + t or value >= hi - t
    )
    if at_bound:
        advice = []
        if any(not name.startswith("shift") for name in at_bound):
            advice.append("consider fewer layers")
        if any(name.startswith("shift") for name in at_bound):
            advice.append("a static shift beyond a factor of 2 suggests a data problem")
        warnings.warn(
            f"parameters at their search bounds, not resolved by the data: "
            f"{', '.join(at_bound)}; {'; '.join(advice)}",
            DataWarning,
            stacklevel=2,
        )

    chi2 = float(2.0 * best.cost)
    dof = data.n_data - problem.n_params
    p_value = float(stats.chi2.sf(chi2, dof))
    if p_value < POOR_FIT_P_VALUE:
        warnings.warn(
            f"poor fit: chi2 = {chi2:.1f} for {dof} degrees of freedom "
            f"(p = {p_value:.2g}); the errors may be underestimated, or more "
            "layers may be needed",
            DataWarning,
            stacklevel=2,
        )

    weighted = problem.jacobian(m) / problem.s[:, np.newaxis]
    covariance, std_log, correlation = _linearized_covariance(weighted)
    resistivity, thickness = np.exp(m[:n]), np.exp(m[n : 2 * n - 1])
    # Depth z_i = h_1 + ... + h_i; its gradient with respect to ln h_j is h_j (j <= i).
    grad = np.tril(np.ones((n - 1, n - 1))) * thickness[np.newaxis, :]
    h_std = std_log[n : 2 * n - 1]
    finite_h = np.isfinite(h_std)
    cov_h = np.where(
        np.outer(finite_h, finite_h), covariance[n : 2 * n - 1, n : 2 * n - 1], 0.0
    )
    depth_std = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", grad, cov_h, grad), 0.0, None))
    depth_std[np.cumsum(~finite_h) > 0] = np.inf
    shifts = np.exp(np.concatenate([[0.0], m[2 * n - 1 :]]))
    if not problem.n_shifts:
        shifts = np.ones(int(problem.segment.max()) + 1)

    return LayerFit(
        predicted=np.exp(problem.predict(m).real),
        chi2=chi2,
        chi2_target=float(dof),
        n_data=data.n_data,
        converged=converged,
        stop_reason=str(best.message),
        n_iterations=int(best.nfev),
        model=LayeredModel(resistivity=resistivity, thickness=thickness),
        parameter_names=names,
        std_log=std_log,
        covariance=covariance,
        correlation=correlation,
        depths=np.cumsum(thickness),
        depth_std=depth_std,
        static_shift=shifts,
        p_value=p_value,
        n_starts=len(starts),
        at_bound=at_bound,
    )
