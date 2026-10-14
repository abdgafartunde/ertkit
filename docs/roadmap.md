# Roadmap

Each release is complete, verified and reviewed before it is published. The
[design document](dev/design.md) states the acceptance criteria for each one.

| Release | Content |
|---|---|
| 0.1 | VES: layered-earth forward model; few-layer inversion with linearized uncertainty; smooth (Occam) inversion with sensitivity and 1-D depth of investigation; CSV input. First PyPI release. |
| 0.2 | VES equivalence analysis by Markov chain Monte Carlo, with convergence diagnostics; comparison of the number of layers. |
| 0.3 | 2.5-D ERT forward modeling: finite elements, standard arrays, pseudosections. |
| 0.4 | ERT inversion with an error model and a $\chi^2$ target; adjoint sensitivities; cumulative sensitivity. |
| 0.5 | ERT depth-of-investigation index and uncertainty maps, used to mask unresolved regions by default. |
| 0.6 | RES2DINV and pyGIMLi data formats; reciprocal-error analysis; one open field dataset end to end. |
| 1.0 | Stable public API; documentation of every public function and of the theory. |

Not planned for 1.0: 3-D surveys, surface topography, time-lapse inversion,
induced polarization, joint inversion, machine learning, and the
petrophysical translation of resistivity into water content.
