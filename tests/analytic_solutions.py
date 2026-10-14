"""Closed-form reference solutions used by the verification tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def two_layer_potential(
    r: ArrayLike, rho1: float, rho2: float, h: float, current: float = 1.0
) -> NDArray[np.float64]:
    r"""Surface potential over a two-layer earth by the method of images.

    .. math::

        V(r) = \frac{I\rho_1}{2\pi r} \left[ 1 + 2 \sum_{n=1}^{\infty}
               \frac{\kappa^n}{\sqrt{1 + (2nh/r)^2}} \right],
        \qquad \kappa = \frac{\rho_2 - \rho_1}{\rho_2 + \rho_1}.

    The series is summed until :math:`|\kappa|^n < 10^{-14}`; the neglected
    tail is then below :math:`10^{-14}/(1 - |\kappa|)` relative to the leading
    term, about :math:`10^{-11}` for the largest contrast used (1000).
    """
    r = np.atleast_1d(np.asarray(r, dtype=np.float64))
    kappa = (rho2 - rho1) / (rho2 + rho1)
    total = np.ones_like(r)
    if kappa != 0.0:
        n_terms = int(np.ceil(np.log(1e-14) / np.log(abs(kappa))))
        n = np.arange(1, n_terms + 1, dtype=np.float64)
        terms = kappa ** n[np.newaxis, :] / np.sqrt(
            1.0 + (2.0 * n[np.newaxis, :] * h / r[:, np.newaxis]) ** 2
        )
        total = total + 2.0 * terms.sum(axis=1)
    return np.asarray(current * rho1 / (2.0 * np.pi * r) * total)


def lipschitz_integral(a: float, r: ArrayLike) -> NDArray[np.float64]:
    r""":math:`\int_0^\infty e^{-a\lambda} J_0(\lambda r)\,d\lambda = 1/\sqrt{a^2 + r^2}`."""
    r = np.asarray(r, dtype=np.float64)
    return np.asarray(1.0 / np.sqrt(a * a + r * r))
