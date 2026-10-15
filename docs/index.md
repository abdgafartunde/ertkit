# ertkit

**Resistivity imaging for groundwater exploration, with an honest account of
what the data can resolve.**

ertkit is a Python package, in early development, for inverting vertical
electrical sounding (VES) data, with 2.5-D electrical resistivity tomography
(ERT) to follow. Each result comes with a data error model and a fit to the
noise level, sensitivity and depth of investigation, and parameter
uncertainty; from release 0.2, VES results are planned to include the range of
equivalent models.

Boreholes are often sited from resistivity models whose reliability was never
quantified. Open tools such as [pyGIMLi](https://www.pygimli.org),
[SimPEG](https://simpeg.xyz) and [ResIPy](https://gitlab.com/hkex/resipy)
already provide excellent modeling and inversion; ertkit focuses on making
reliability the default output, publishing a verification report with every
release, and keeping the code small enough to teach from.

The first release, 0.1, covers vertical electrical sounding: a layered-earth
forward model, few-layer inversion with linearized uncertainty, smooth
inversion with a depth of investigation, and CSV input. Start with the
[examples](auto_examples/index.rst) and the
[user guide](user_guide/ves.md); the [verification report](verification.md)
lists the checks behind the results, and the [roadmap](roadmap.md) the
releases to come.

```{toctree}
:maxdepth: 1
:caption: User guide

install
user_guide/ves
auto_examples/index
verification
roadmap
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
changelog
citing
```

```{toctree}
:maxdepth: 1
:caption: Development

contributing
dev/index
```
