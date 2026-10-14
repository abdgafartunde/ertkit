# ertkit

**Resistivity imaging for groundwater exploration, with an honest account of
what the data can resolve.**

ertkit is a Python package, in early development, for inverting vertical
electrical sounding (VES) and 2.5-D electrical resistivity tomography (ERT)
data. Each result is designed to come with a data error model and a fit to the
noise level, sensitivity and depth of investigation, and parameter
uncertainty, including, for VES, the range of equivalent models.

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
