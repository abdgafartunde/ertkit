"""Observed or synthetic VES data."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ertkit._warnings import DataWarning
from ertkit.noise import ErrorModel
from ertkit.ves.forward import forward
from ertkit.ves.model import LayeredModel
from ertkit.ves.survey import Survey

__all__ = ["Sounding"]


@dataclass(frozen=True, eq=False, kw_only=True, slots=True)
class Sounding:
    r"""Apparent resistivities of a sounding with their error model.

    Readings that cannot be inverted in log space (transfer resistance
    :math:`R \le 0`, or :math:`|R|` below three times the absolute error) are
    removed with a :class:`~ertkit.DataWarning`.

    Parameters
    ----------
    survey : Survey
        Electrode half-spacings of the readings.
    rhoa : array_like
        Apparent resistivity in ohm meters, one value per reading.
    error : ErrorModel, optional
        Measurement-error model. Default ``ErrorModel()`` (3% relative).

    Raises
    ------
    ValueError
        If ``rhoa`` does not have one finite value per reading, or if no
        reading is usable.
    """

    survey: Survey
    rhoa: NDArray[np.float64]
    error: ErrorModel = field(default_factory=ErrorModel)

    def __post_init__(self) -> None:
        rhoa = np.array(self.rhoa, dtype=np.float64, copy=True, ndmin=1)
        if rhoa.shape != (self.survey.n_readings,):
            raise ValueError(
                f"need {self.survey.n_readings} apparent resistivities, got shape {rhoa.shape}"
            )
        if not np.all(np.isfinite(rhoa)):
            raise ValueError("apparent resistivities must be finite")
        keep = self.error.usable(rhoa / self.survey.geometric_factor)
        if not np.any(keep):
            raise ValueError(
                "no reading is usable (all R <= 0 or below the noise level)"
            )
        survey = self.survey
        if not np.all(keep):
            dropped = np.flatnonzero(~keep).tolist()
            warnings.warn(
                f"removed {len(dropped)} reading(s) with R <= 0 or |R| below three "
                f"times the absolute error: indices {dropped}",
                DataWarning,
                stacklevel=3,
            )
            survey = Survey(ab2=survey.ab2[keep], mn2=survey.mn2[keep])
            rhoa = rhoa[keep]
        rhoa.setflags(write=False)
        object.__setattr__(self, "survey", survey)
        object.__setattr__(self, "rhoa", rhoa)

    @classmethod
    def synthetic(
        cls,
        model: LayeredModel,
        survey: Survey,
        error: ErrorModel | None = None,
        rng: int | np.random.Generator | None = None,
    ) -> Sounding:
        r"""Simulate noisy data from a layered model.

        Noise is drawn from the error model in log space:
        :math:`\ln \rho_a^{\mathrm{obs}} = \ln \rho_a + s \, \varepsilon`, with
        :math:`\varepsilon \sim N(0, 1)` and :math:`s` evaluated at the
        noise-free data.

        Parameters
        ----------
        model : LayeredModel
            The true model.
        survey : Survey
            Electrode half-spacings.
        error : ErrorModel, optional
            Error model used both to draw the noise and to weight the data.
            Default ``ErrorModel()``.
        rng : int, numpy.random.Generator or None, optional
            Seed or generator, normalized with :func:`numpy.random.default_rng`.

        Returns
        -------
        Sounding
        """
        error = ErrorModel() if error is None else error
        generator = np.random.default_rng(rng)
        clean = forward(model, survey)
        s = error.log_std(clean / survey.geometric_factor)
        noisy = clean * np.exp(s * generator.standard_normal(clean.size))
        return cls(survey=survey, rhoa=noisy, error=error)

    @property
    def n_data(self) -> int:
        """Number of readings."""
        return int(self.rhoa.size)

    @property
    def resistance(self) -> NDArray[np.float64]:
        r"""Transfer resistance :math:`R = \rho_a / K`, in ohms."""
        return self.rhoa / self.survey.geometric_factor

    @property
    def log_data(self) -> NDArray[np.float64]:
        r"""The inverted data :math:`d = \ln \rho_a`."""
        return np.log(self.rhoa)

    @property
    def log_std(self) -> NDArray[np.float64]:
        r"""Standard deviation of :math:`\ln \rho_a` for each reading."""
        return self.error.log_std(self.resistance)
