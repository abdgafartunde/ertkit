"""Layered-earth models for vertical electrical sounding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["LayeredModel"]


def _readonly(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.array(values, dtype=np.float64, copy=True, ndmin=1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"{name} must be positive and finite")
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False, kw_only=True, slots=True)
class LayeredModel:
    r"""A horizontally layered earth.

    Layer :math:`i` has resistivity :math:`\rho_i`; layers
    :math:`1, \dots, N-1` have thickness :math:`h_i`, and layer :math:`N`
    is a half-space.

    Parameters
    ----------
    resistivity : array_like
        Layer resistivities in ohm meters, from the top; length :math:`N`.
    thickness : array_like
        Thicknesses in meters of the upper :math:`N - 1` layers. Empty for a
        homogeneous half-space.

    Raises
    ------
    ValueError
        If a value is not positive and finite, or if there is not exactly one
        thickness fewer than resistivities.

    Examples
    --------
    >>> from ertkit.ves import LayeredModel
    >>> model = LayeredModel(resistivity=[150.0, 30.0, 800.0], thickness=[2.0, 15.0])
    >>> model.n_layers
    3
    >>> model.depths.tolist()
    [2.0, 17.0]
    """

    resistivity: NDArray[np.float64]
    thickness: NDArray[np.float64]

    def __post_init__(self) -> None:
        rho = _readonly(self.resistivity, "resistivity")
        h = _readonly(self.thickness, "thickness")
        if h.size != rho.size - 1:
            raise ValueError(
                f"need {rho.size - 1} thicknesses for {rho.size} layers, got {h.size}"
            )
        object.__setattr__(self, "resistivity", rho)
        object.__setattr__(self, "thickness", h)

    @property
    def n_layers(self) -> int:
        """Number of layers, including the bottom half-space."""
        return int(self.resistivity.size)

    @property
    def depths(self) -> NDArray[np.float64]:
        """Depths in meters of the bottoms of the upper :math:`N - 1` layers."""
        return np.cumsum(self.thickness)

    @property
    def conductance(self) -> NDArray[np.float64]:
        r"""Longitudinal conductance :math:`S_i = h_i/\rho_i` of the upper layers, in siemens."""
        return self.thickness / self.resistivity[:-1]

    @property
    def transverse_resistance(self) -> NDArray[np.float64]:
        r"""Transverse resistance :math:`T_i = h_i \rho_i` of the upper layers, in ohm square meters."""
        return self.thickness * self.resistivity[:-1]
