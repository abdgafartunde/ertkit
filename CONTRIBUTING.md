# Contributing to ertkit

Thank you for your interest in ertkit. Bug reports, questions, suggestions and
pull requests are all welcome. Everyone taking part follows the
[Code of Conduct](https://github.com/abdgafartunde/ertkit/blob/main/CODE_OF_CONDUCT.md).

## Support

ertkit is maintained by one researcher on a best-effort basis. Questions and
bug reports go to the [issue tracker](https://github.com/abdgafartunde/ertkit/issues);
a bug report should include the output of `ertkit.show_versions()`.

## Development setup

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/abdgafartunde/ertkit.git
cd ertkit
uv sync                          # creates .venv with all development tools
uv run pre-commit install        # runs the formatters and linters on commit
```

## Everyday commands

| Task | Command |
|---|---|
| Tests | `uv run pytest` |
| Tests with coverage | `uv run pytest --cov` |
| Fast tests only | `uv run pytest -m "not slow"` |
| Lint and format | `uv run pre-commit run --all-files` |
| Type check | `uv run mypy` |
| Verification report (for the docs) | `uv run pytest -m verification --verification-report=docs/_generated/verification.json` |
| Documentation | `uv run sphinx-build -W --keep-going docs docs/_build/html` |

Test markers: `slow` (long tests, skipped by the oldest-dependencies job),
`verification` (checks that feed the verification report) and `crosscheck`
(comparisons with pyGIMLi and SimPEG, run weekly).

## Pull requests

One pull request per feature or fix. Each pull request:

- adds or updates tests, and states the justification for every numerical
  tolerance next to it;
- updates the documentation and adds an entry under `[Unreleased]` in
  `CHANGELOG.md`;
- passes all CI checks before it is merged (merges are squash merges).

Numbers in the documentation are never typed by hand: they come from the
example gallery, the verification report or named constants.

## Code provenance

No code is adapted from GPL-licensed projects such as ResIPy. Code adapted from
permissively licensed projects (for example pyGIMLi, Apache-2.0) keeps their
attribution. Digital filter coefficients come from libdlf, whose filter values
are licensed CC BY 4.0 and are attributed in the documentation.

## Use of AI tools

Development uses AI assistance: Claude Code (Anthropic) helps write code,
tests and documentation, and runs reviews of the mathematics, the code and the
documentation before releases. The maintainer makes the design decisions and
personally checks derivations, tolerances and the verification report before
each release. Each release's notes summarize the AI assistance and what was
checked by hand.

## Documentation deployment

The documentation is built for every pull request and push to `main`, and
deployed when a release is published, so the website matches the latest
release; a manual run of the Docs workflow redeploys it (for example after
adding the DOI).

## Maintainer's sign-off checklist

Before each release the maintainer checks, personally:

- the derivations behind the release's new methods, against the design
  document and the cited papers;
- that every tolerance in the verification report has a sound justification;
- the verification report and the cross-check results of the release
  candidate;
- that the examples' text matches what they print;
- the AI-assisted review's findings and how each was resolved.

## Release checklist

1. All CI checks pass on `main`, and the weekly cross-check workflow has been
   run on the release candidate.
2. The AI-assisted review is done and the maintainer's sign-off checklist
   above is complete.
3. `CHANGELOG.md`: rename `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD` and add a
   new empty `[Unreleased]` section.
4. `CITATION.cff`: set `version` and `date-released`. `README.md`: update the
   status section.
5. `uv run pre-commit autoupdate` has been run, unless pre-commit.ci keeps
   the hooks current.
6. A development build uploaded to TestPyPI (manual run of the `Release`
   workflow) installs and imports correctly.
7. Check that `.zenodo.json` is present, the Zenodo webhook is on, the PyPI
   trusted publisher exists, and the `github-pages` environment allows tags
   `v*`.
8. Create the GitHub release with tag `vX.Y.Z`; its notes summarize the AI
   assistance and what the maintainer checked by hand. The `Release` workflow
   checks that the tag matches `CITATION.cff` and `CHANGELOG.md`, then uploads
   to PyPI; Zenodo archives the release and issues a DOI, and the Docs
   workflow deploys the documentation.
9. After the release: add the DOI to `CITATION.cff`, the README and
   `docs/citing.md`, then run the Docs workflow manually.

## Repository review exceptions

The repository is checked with the Scientific Python
[repo-review](https://learn.scientific-python.org/development/guides/repo-review/).
Three checks are deliberately not followed:

- **PY007 (task runner).** The uv commands above play this role; a nox or tox
  configuration would duplicate them.
- **PC140 (type checker in pre-commit).** mypy runs as its own CI step from
  the locked environment, where it sees numpy, scipy and their type stubs; in
  pre-commit's isolated environment it would not.
- **PC180 (Markdown formatter).** Markdown files contain LaTeX math that
  formatters can rewrite in ways GitHub then renders wrongly.
