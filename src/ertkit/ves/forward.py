r"""Forward modeling of vertical electrical soundings over a layered earth.

The potential on the surface at distance :math:`r` from a point source of
current :math:`I` is

.. math::

    V(r) = \frac{I}{2\pi} \int_0^\infty \mathcal{T}_1(\lambda) J_0(\lambda r) \, d\lambda,

where :math:`\mathcal{T}_1` is the resistivity transform of the layered earth
(Pekeris 1940; Koefoed 1979). The integral is evaluated with the 120-point
digital linear filter of Guptasarma and Singh (1997), taken from libdlf (filter
values licensed CC BY 4.0).

The functions here accept complex resistivities and thicknesses, so that
derivatives can be computed by the complex-step method.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

import libdlf
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ertkit.ves.model import LayeredModel
from ertkit.ves.survey import Survey

__all__ = ["forward", "hankel_j0", "potential", "resistivity_transform"]


@cache
def _filter() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Base and J0 weights of the Guptasarma and Singh (1997) 120-point filter."""
    base, weights = libdlf.hankel.gupt_120_1997()
    return np.asarray(base, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def resistivity_transform(
    lam: ArrayLike, resistivity: ArrayLike, thickness: ArrayLike
) -> NDArray[Any]:
    r"""Resistivity transform :math:`\mathcal{T}_1(\lambda)` of a layered earth.

    Computed by the recurrence

    .. math::

        \mathcal{T}_N = \rho_N, \qquad
        \mathcal{T}_i = \frac{\mathcal{T}_{i+1} + \rho_i \tanh(\lambda h_i)}
                             {1 + \mathcal{T}_{i+1} \tanh(\lambda h_i)/\rho_i},
        \qquad i = N-1, \dots, 1.

    Parameters
    ----------
    lam : array_like
        Hankel-transform variable :math:`\lambda`, in 1/m; any shape.
    resistivity : array_like
        Layer resistivities in ohm meters, length :math:`N`.
    thickness : array_like
        Thicknesses in meters of the upper :math:`N - 1` layers.

    Returns
    -------
    ndarray
        :math:`\mathcal{T}_1(\lambda)` in ohm meters, with the shape of ``lam``.
        Complex if any input is complex.
    """
    rho = np.asarray(resistivity)
    h = np.asarray(thickness)
    lam = np.asarray(lam)
    out = np.broadcast_to(rho[-1], lam.shape).astype(np.result_type(lam, rho, h, 1.0))
    for i in range(rho.size - 2, -1, -1):
        t = np.tanh(lam * h[i])
        out = (out + rho[i] * t) / (1.0 + out * t / rho[i])
    return out


def hankel_j0(
    kernel: Callable[[NDArray[np.float64]], NDArray[Any]], r: ArrayLike
) -> NDArray[Any]:
    r"""Zeroth-order Hankel transform by digital linear filtering.

    Approximates

    .. math::

        F(r) = \int_0^\infty f(\lambda) J_0(\lambda r) \, d\lambda
             \approx \frac{1}{r} \sum_j f(b_j / r) \, w_j,

    with the base :math:`b_j` and weights :math:`w_j` of the 120-point filter
    of Guptasarma and Singh (1997).

    Parameters
    ----------
    kernel : callable
        The function :math:`f`. It receives an array of :math:`\lambda` values
        of shape ``(len(r), 120)`` and must return an array of the same shape.
    r : array_like
        Positive distances in meters, one-dimensional.

    Returns
    -------
    ndarray
        :math:`F(r)`, one value per distance.
    """
    base, weights = _filter()
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    lam = base[np.newaxis, :] / r[:, np.newaxis]
    values = np.asarray(kernel(lam))
    return np.asarray(values @ weights / r)


def _rhoa(
    resistivity: NDArray[Any],
    thickness: NDArray[Any],
    ab2: NDArray[np.float64],
    mn2: NDArray[np.float64],
) -> NDArray[Any]:
    """Apparent resistivity of symmetric soundings; complex inputs allowed."""
    near = ab2 - mn2
    far = ab2 + mn2
    r = np.concatenate([near, far])
    v = hankel_j0(lambda lam: resistivity_transform(lam, resistivity, thickness), r) / (
        2.0 * np.pi
    )
    v_near, v_far = v[: ab2.size], v[ab2.size :]
    # Delta V = V(AM) - V(BM) - V(AN) + V(BN) = 2 [V(near) - V(far)] for I = 1 A.
    k = 2.0 * np.pi / (2.0 / near - 2.0 / far)
    return np.asarray(k * 2.0 * (v_near - v_far))


def potential(
    model: LayeredModel, r: ArrayLike, current: float = 1.0
) -> NDArray[np.float64]:
    """Surface potential at distance ``r`` from a point current source.

    Parameters
    ----------
    model : LayeredModel
        The layered earth.
    r : array_like
        Distances in meters from the source, positive.
    current : float, optional
        Source current in amperes. Default 1.

    Returns
    -------
    ndarray of float
        Potential in volts, one value per distance.
    """
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    if np.any(~(r > 0)) or not np.all(np.isfinite(r)):
        raise ValueError("distances must be positive and finite")
    integral = hankel_j0(
        lambda lam: resistivity_transform(lam, model.resistivity, model.thickness), r
    )
    return np.asarray(current * integral.real / (2.0 * np.pi), dtype=np.float64)


def forward(model: LayeredModel, survey: Survey) -> NDArray[np.float64]:
    """Apparent resistivities of a sounding over a layered earth.

    Parameters
    ----------
    model : LayeredModel
        The layered earth.
    survey : Survey
        The readings, with their half-spacings.

    Returns
    -------
    ndarray of float
        Apparent resistivity in ohm meters, one value per reading. The finite
        potential-electrode spacing is modeled exactly.

    Examples
    --------
    >>> import numpy as np
    >>> from ertkit.ves import LayeredModel, Survey, forward
    >>> survey = Survey.schlumberger(ab2=[3.0, 30.0, 300.0], mn2=0.5)
    >>> model = LayeredModel(resistivity=[100.0], thickness=[])
    >>> np.allclose(forward(model, survey), 100.0)
    True
    """
    rhoa = _rhoa(model.resistivity, model.thickness, survey.ab2, survey.mn2)
    return np.asarray(rhoa.real, dtype=np.float64)
