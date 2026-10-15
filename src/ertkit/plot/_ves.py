"""Plots for vertical electrical soundings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ertkit.results import InversionResult
from ertkit.ves.data import Sounding
from ertkit.ves.invert import LayerFit, _segments
from ertkit.ves.model import LayeredModel
from ertkit.ves.smooth import DOI_THRESHOLD, SmoothFit

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = ["layered_model", "sounding"]

_MAX_FACTOR = 100.0
"""Resistivity error bars are drawn to at most this factor either side."""


def _axes(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    _, new = plt.subplots(figsize=(6.0, 4.5), layout="constrained")
    return new


def sounding(
    data: Sounding,
    *fits: InversionResult,
    labels: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot apparent resistivity against AB/2, with the data errors and fitted curves.

    Fitted curves are drawn separately for each MN segment (readings with the
    same MN/2), so that static shifts between segments show as steps. When
    MN changes with every reading, as in a Wenner sounding, one curve is drawn.

    Parameters
    ----------
    data : Sounding
        The observed data; error bars show one standard deviation.
    *fits : InversionResult
        Inversion results whose predicted data are drawn as lines.
    labels : sequence of str, optional
        One label per fit. Default "fit 1", "fit 2", and so on.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is made if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _axes(ax)
    ab2, rhoa, s = data.survey.ab2, data.rhoa, data.log_std
    yerr = np.vstack([rhoa * (1.0 - np.exp(-s)), rhoa * (np.exp(s) - 1.0)])
    ax.errorbar(ab2, rhoa, yerr=yerr, fmt="o", ms=4, color="black", label="data")
    labels = (
        [f"fit {i + 1}" for i in range(len(fits))] if labels is None else list(labels)
    )
    segment = _segments(data.survey.mn2)
    if np.bincount(segment).max() < 2:
        segment = np.zeros_like(segment)
    for fit, label in zip(fits, labels, strict=True):
        color = None
        for k in np.unique(segment):
            # Readings of one MN segment, in order of AB/2.
            sel = np.flatnonzero(segment == k)
            sel = sel[np.argsort(ab2[sel], kind="stable")]
            (line,) = ax.plot(
                ab2[sel],
                fit.predicted[sel],
                color=color,
                label=label if k == segment[0] else None,
            )
            color = line.get_color()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("AB/2 (m)")
    ax.set_ylabel(r"apparent resistivity ($\Omega$ m)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax


def _steps(
    model: LayeredModel, depth_max: float, depth_min: float = 0.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resistivity and depth arrays for a step plot from depth_min to depth_max."""
    tops = np.concatenate([[depth_min], model.depths])
    bottoms = np.concatenate([model.depths, [max(depth_max, tops[-1])]])
    depth = np.ravel(np.column_stack([tops, bottoms]))
    rho = np.repeat(model.resistivity, 2)
    return rho, depth


def _default_depth_max(
    items: Sequence[LayeredModel | LayerFit | SmoothFit], models: list[LayeredModel]
) -> float:
    """1.5 times the deepest depth of investigation, else the deepest interface."""
    dois = [
        item.doi
        for item in items
        if isinstance(item, SmoothFit) and np.isfinite(item.doi) and item.doi > 0
    ]
    interfaces = [
        float(m.depths[-1])
        for item, m in zip(items, models, strict=True)
        if m.depths.size and not isinstance(item, SmoothFit)
    ]
    return 1.5 * max(dois or interfaces or [10.0])


def layered_model(
    *items: LayeredModel | LayerFit | SmoothFit,
    labels: Sequence[str] | None = None,
    depth_max: float | None = None,
    log_depth: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot layered models as resistivity against depth.

    Few-layer fits show one standard deviation of each layer's resistivity
    and of each interface depth; error bars are cut at the edges of the plot,
    and resistivity error bars at a factor of 100 (``LayerFit.factor`` gives
    the values). Smooth fits shade the region below their depth of
    investigation, where the model reflects the regularization rather than the
    data.

    Parameters
    ----------
    *items : LayeredModel, LayerFit or SmoothFit
        Models, or inversion results whose models are drawn.
    labels : sequence of str, optional
        One label per item. Default "model 1", "model 2", and so on.
    depth_max : float, optional
        Deepest depth shown, in meters. Default: 1.5 times the depth of
        investigation of the smooth fits, or else 1.5 times the deepest
        interface.
    log_depth : bool, optional
        Use a logarithmic depth axis, which suits smooth models whose layers
        thicken with depth. Default False.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is made if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _axes(ax)
    labels = (
        [f"model {i + 1}" for i in range(len(items))]
        if labels is None
        else list(labels)
    )
    models = [item if isinstance(item, LayeredModel) else item.model for item in items]
    if depth_max is None:
        depth_max = _default_depth_max(items, models)
    shallowest = [float(m.depths[0]) for m in models if m.depths.size]
    depth_min = 0.5 * min(shallowest, default=1.0) if log_depth else 0.0
    for item, model, label in zip(items, models, labels, strict=True):
        rho, depth = _steps(model, depth_max, depth_min)
        (line,) = ax.plot(rho, depth, label=label)
        if isinstance(item, LayerFit):
            n = model.n_layers
            tops = np.concatenate([[depth_min], model.depths])
            bottoms = np.concatenate([model.depths, [depth_max]])
            mid = np.sqrt(tops * bottoms) if log_depth else 0.5 * (tops + bottoms)
            factor = np.minimum(item.factor[:n], _MAX_FACTOR)
            ax.errorbar(
                model.resistivity,
                mid,
                xerr=[
                    model.resistivity * (1 - 1 / factor),
                    model.resistivity * (factor - 1),
                ],
                fmt="none",
                color=line.get_color(),
                capsize=3,
            )
            if n > 1:
                ax.errorbar(
                    np.sqrt(model.resistivity[:-1] * model.resistivity[1:]),
                    model.depths,
                    # Error bars stop at the edges of the plot.
                    yerr=[
                        np.minimum(item.depth_std, model.depths - depth_min).clip(0),
                        np.minimum(item.depth_std, depth_max - model.depths).clip(0),
                    ],
                    fmt="none",
                    color=line.get_color(),
                    capsize=3,
                )
        if (
            isinstance(item, SmoothFit)
            and np.isfinite(item.doi)
            and item.doi < depth_max
        ):
            ax.axhspan(
                max(item.doi, depth_min),
                depth_max,
                color=line.get_color(),
                alpha=0.12,
                label=f"below depth of investigation ({label})",
            )
            # Isolated layers above the depth of investigation whose index
            # also reaches the threshold are unresolved too.
            tops = np.concatenate([[0.0], model.depths])
            bottoms = np.concatenate([model.depths, [depth_max]])
            for j in np.flatnonzero(item.doi_index >= DOI_THRESHOLD):
                if bottoms[j] <= item.doi:
                    ax.axhspan(
                        max(tops[j], depth_min),
                        bottoms[j],
                        color=line.get_color(),
                        alpha=0.12,
                    )
    ax.set_xscale("log")
    if log_depth:
        ax.set_yscale("log")
    ax.set_ylim(depth_max, depth_min)
    ax.set_xlabel(r"resistivity ($\Omega$ m)")
    ax.set_ylabel("depth (m)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax
