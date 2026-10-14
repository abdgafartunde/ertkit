"""Symmetric four-electrode soundings (Schlumberger and Wenner)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ertkit.arrays import geometric_factor

__all__ = ["Survey"]


@dataclass(frozen=True, eq=False, kw_only=True, slots=True)
class Survey:
    r"""Readings of a symmetric collinear sounding.

    The current electrodes sit at :math:`\pm AB/2` and the potential
    electrodes at :math:`\pm MN/2` about the sounding center, so each reading
    is given by its half-spacings. Schlumberger soundings with a changing
    :math:`MN` and Wenner soundings (:math:`MN/2 = AB/6`) are both covered.

    Parameters
    ----------
    ab2 : array_like
        Half the current-electrode spacing, :math:`AB/2`, in meters, one value
        per reading.
    mn2 : array_like
        Half the potential-electrode spacing, :math:`MN/2`, in meters: a single
        value for all readings or one per reading.

    Raises
    ------
    ValueError
        If spacings are not positive and finite, or if :math:`MN/2 \ge AB/2`
        for any reading.

    See Also
    --------
    Survey.schlumberger, Survey.wenner

    Examples
    --------
    >>> from ertkit.ves import Survey
    >>> survey = Survey.schlumberger(ab2=[3.0, 10.0, 30.0], mn2=0.5)
    >>> survey.n_readings
    3
    >>> Survey.wenner([2.0, 5.0]).mn2.tolist()
    [1.0, 2.5]
    """

    ab2: NDArray[np.float64]
    mn2: NDArray[np.float64]

    def __post_init__(self) -> None:
        ab2 = np.array(self.ab2, dtype=np.float64, copy=True, ndmin=1)
        mn2 = np.array(
            np.broadcast_to(np.asarray(self.mn2, dtype=np.float64), ab2.shape),
            dtype=np.float64,
            copy=True,
        )
        if ab2.ndim != 1 or ab2.size == 0:
            raise ValueError("ab2 must be a non-empty one-dimensional array")
        for name, arr in (("ab2", ab2), ("mn2", mn2)):
            if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
                raise ValueError(f"{name} must be positive and finite")
        if np.any(mn2 >= ab2):
            raise ValueError("MN/2 must be smaller than AB/2 for every reading")
        ab2.setflags(write=False)
        mn2.setflags(write=False)
        object.__setattr__(self, "ab2", ab2)
        object.__setattr__(self, "mn2", mn2)

    @classmethod
    def schlumberger(cls, ab2: ArrayLike, mn2: ArrayLike) -> Survey:
        """Create a Schlumberger sounding.

        Parameters
        ----------
        ab2, mn2 : array_like
            Half-spacings in meters, as in :class:`Survey`.

        Returns
        -------
        Survey
        """
        return cls(
            ab2=np.asarray(ab2, dtype=np.float64), mn2=np.asarray(mn2, dtype=np.float64)
        )

    @classmethod
    def wenner(cls, a: ArrayLike) -> Survey:
        """Create a Wenner sounding with electrode spacings ``a``.

        Parameters
        ----------
        a : array_like
            Spacing between adjacent electrodes, in meters, one per reading.

        Returns
        -------
        Survey
            Readings with :math:`AB/2 = 1.5a` and :math:`MN/2 = 0.5a`.
        """
        a = np.asarray(a, dtype=np.float64)
        return cls(ab2=1.5 * a, mn2=0.5 * a)

    @property
    def n_readings(self) -> int:
        """Number of readings."""
        return int(self.ab2.size)

    @property
    def geometric_factor(self) -> NDArray[np.float64]:
        r"""Geometric factor :math:`K` of each reading, in meters."""
        near = self.ab2 - self.mn2
        far = self.ab2 + self.mn2
        return geometric_factor(near, far, far, near)
