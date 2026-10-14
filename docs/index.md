# ertkit

**Resistivity imaging for groundwater exploration, with an honest account of
what the data can resolve.**

ertkit is a Python package, in early development, for inverting vertical
electrical sounding (VES) and 2.5-D electrical resistivity tomography (ERT)
data. Each result is designed to come with a data error model and a fit to the
noise level, sensitivity and depth of investigation, and parameter
uncertainty, including, for VES, the range of equivalent models.

The project is at its foundation stage: there is no science code yet. The
[roadmap](roadmap.md) lists the planned releases, and the
[design document](dev/design.md) explains the mathematics and the
verification plan.

```{toctree}
:maxdepth: 1
:caption: User guide

install
roadmap
auto_examples/index
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
