"""Electrode configurations and geometric factors.

All distances are in meters. A remote (infinitely distant) electrode is
given as ``np.inf``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["geometric_factor"]


def geometric_factor(
    am: ArrayLike, bm: ArrayLike, an: ArrayLike, bn: ArrayLike
) -> NDArray[np.float64]:
    r"""Geometric factor of a four-electrode array on a half-space.

    For current electrodes :math:`A`, :math:`B` and potential electrodes
    :math:`M`, :math:`N` on the surface of a homogeneous half-space,

    .. math::

        K = \frac{2\pi}{1/AM - 1/BM - 1/AN + 1/BN},

    so that the apparent resistivity is :math:`\rho_a = K \Delta V / I` with
    :math:`\Delta V = V_M - V_N`.

    Parameters
    ----------
    am, bm, an, bn : array_like
        Distances in meters between the electrodes named by each argument.
        They broadcast against each other. Use ``np.inf`` for a remote
        electrode (for example ``bm = bn = np.inf`` for a pole-dipole array).

    Returns
    -------
    ndarray of float
        Geometric factor in meters. It is negative when :math:`V_M < V_N` on a
        half-space, as in a dipole-dipole array ordered A, B, M, N, where
        :math:`M` is nearer than :math:`N` to the current sink :math:`B`.

    Raises
    ------
    ValueError
        If a distance is not positive, or if the configuration measures no
        potential difference on a half-space (zero denominator).

    Examples
    --------
    A Wenner array with spacing 10 m:

    >>> import numpy as np
    >>> from ertkit.arrays import geometric_factor
    >>> float(geometric_factor(10.0, 20.0, 20.0, 10.0) / (2 * np.pi))
    10.0
    """
    dists = np.broadcast_arrays(
        *(np.asarray(d, dtype=np.float64) for d in (am, bm, an, bn))
    )
    if any(np.any(~(d > 0)) for d in dists):
        raise ValueError("electrode distances must be positive (np.inf for remote)")
    d_am, d_bm, d_an, d_bn = dists
    denom = 1.0 / d_am - 1.0 / d_bm - 1.0 / d_an + 1.0 / d_bn
    scale = 1.0 / np.minimum.reduce(dists)
    if np.any(np.abs(denom) <= 1e-12 * scale):
        raise ValueError(
            "the configuration measures no potential difference on a half-space"
        )
    return np.asarray(2.0 * np.pi / denom, dtype=np.float64)
