# ertkit

[![CI](https://github.com/abdgafartunde/ertkit/actions/workflows/ci.yml/badge.svg)](https://github.com/abdgafartunde/ertkit/actions/workflows/ci.yml)
[![Docs](https://github.com/abdgafartunde/ertkit/actions/workflows/docs.yml/badge.svg)](https://abdgafartunde.github.io/ertkit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Resistivity imaging for groundwater exploration, with an honest account of
what the data can resolve.**

ertkit is a Python package, in early development, for inverting vertical
electrical sounding (VES) and 2.5-D electrical resistivity tomography (ERT)
data. Each result is designed to come with what is needed to judge it:

- a data error model, and a fit to the noise level rather than to every digit
  of the data;
- sensitivity and depth of investigation, so that unresolved parts of a model
  are marked as such;
- parameter uncertainty and, for VES, the range of equivalent models that fit
  the data equally well.

## Why another resistivity code?

Boreholes are often sited from resistivity models whose reliability was never
quantified. Open tools such as [pyGIMLi](https://www.pygimli.org),
[SimPEG](https://simpeg.xyz) and [ResIPy](https://gitlab.com/hkex/resipy)
already provide excellent modeling and inversion. ertkit focuses on three
things: making reliability the default output, publishing a verification
report with every release, and keeping the code small enough to teach from.
The [design document](docs/dev/design.md) gives the full statement of need,
the mathematics and the verification plan.

## Status

ertkit is in its foundation stage: the repository, tests, documentation and
release pipeline are being set up, and there is no science code yet. Planned
releases:

| Release | Content |
|---|---|
| 0.1 | VES: layered-earth forward model, few-layer and smooth inversion, linearized uncertainty, sensitivity and depth of investigation |
| 0.2 | VES equivalence analysis |
| 0.3 | 2.5-D ERT forward modeling |
| 0.4 | ERT inversion with an error model and cumulative sensitivity |
| 0.5 | ERT depth of investigation and uncertainty maps |
| 0.6 | Field data formats and an end-to-end field example |
| 1.0 | Stable public API |

## Installation

ertkit is not yet on PyPI; release 0.1 will be installable with
`pip install ertkit`. To work on the code, install [uv](https://docs.astral.sh/uv/)
and run:

```bash
git clone https://github.com/abdgafartunde/ertkit.git
cd ertkit
uv sync
uv run pytest
```

## Documentation

The documentation is at <https://abdgafartunde.github.io/ertkit/>. Theory,
examples, the verification report and the API reference are added there with
each release.

## Contributing

Bug reports, questions and suggestions are welcome on the
[issue tracker](https://github.com/abdgafartunde/ertkit/issues). See
[CONTRIBUTING.md](CONTRIBUTING.md) for the development setup and the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Citing

A Zenodo DOI will be issued with release 0.1. Citation metadata is in
[CITATION.cff](CITATION.cff).

## License

MIT; see [LICENSE](LICENSE).
