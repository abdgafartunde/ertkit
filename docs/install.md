# Installation

ertkit needs Python 3.12 or later. Its only dependencies are numpy, scipy and
libdlf; plotting needs matplotlib, which the `plot` extra installs.

## From PyPI

```bash
pip install ertkit            # core
pip install "ertkit[plot]"    # with plotting
```

ertkit's minimum dependency versions are chosen so that it installs in Google
Colab without upgrading the packages Colab provides.

## From source

For development, install [uv](https://docs.astral.sh/uv/) and run:

```bash
git clone https://github.com/abdgafartunde/ertkit.git
cd ertkit
uv sync
uv run pytest
```

## Checking an installation

```python
import ertkit

ertkit.show_versions()
```

Include this output when you report a problem.
