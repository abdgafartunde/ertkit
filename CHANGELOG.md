# Changelog

All notable changes to ertkit are recorded here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and the
project uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Added

- Project foundation (milestone 0.0): packaging with versions from git tags,
  continuous integration on Linux, Windows and macOS, a documentation website,
  a release pipeline using trusted publishing, and contributor guidelines.
- Design document with the mathematics, the verification plan and the
  engineering standards (`docs/dev/design.md`).
- `ertkit.show_versions()` for bug reports.
- VES forward model: `ves.LayeredModel`, `ves.Survey` (Schlumberger soundings
  with finite and changing MN, and Wenner soundings), `ves.forward`,
  `ves.potential`, and the building blocks `ves.resistivity_transform` and
  `ves.hankel_j0` (Guptasarma and Singh 120-point filter).
- `ErrorModel`: measurement errors on the transfer resistance, with a 3%
  relative default that includes a modeling-error floor.
- `ves.Sounding` (data with their error model; readings with R <= 0 or below
  the noise level are removed with a `DataWarning`), synthetic data with
  `Sounding.synthetic`, and CSV input and output with `ves.read_csv` and
  `ves.write_csv` (explicit text encoding).
- Warning categories `ConvergenceWarning` and `DataWarning`.
- Few-layer VES inversion, `ves.invert_few_layer`: log parameters, a
  trust-region least-squares search from several starting models within
  resolvable bounds, complex-step Jacobian, linearized uncertainty
  (standard deviations, multiplicative factors, correlations, depths with
  standard deviations), a goodness-of-fit p-value, optional static-shift
  factors per MN segment, and warnings for poor fits and unresolved
  parameters. Results are `ves.LayerFit` objects.
- Smooth (Occam) VES inversion, `ves.invert_smooth`: many layers with
  logarithmically growing thicknesses, the target misfit chi2 = N_d with the
  largest regularization weight that reaches it, the error-weighted
  sensitivity of each layer, and the Oldenburg and Li depth of investigation
  from two reference-model inversions. Results are `ves.SmoothFit` objects.
- An honest-uncertainty test: over 1000 noise draws, the misfit and the
  coverage of the linearized intervals match their theoretical values.
- `arrays.geometric_factor` for any four-electrode array, including remote
  electrodes.
- Verification tests against the two-layer image series and the Lipschitz
  integral, recorded in a verification report (`pytest -m verification
  --verification-report=PATH`).

[Unreleased]: https://github.com/abdgafartunde/ertkit/commits/main
