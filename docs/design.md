# ertkit design document

Status: draft for review, 14 October 2026. Nothing in this document is
implemented yet. Names in the code sketches may change during review.

## 1. Purpose

ertkit is a Python package for inverting vertical electrical sounding (VES)
and 2.5-D electrical resistivity tomography (ERT) data in groundwater
exploration. Every model it returns comes with an estimate of what the data
can and cannot resolve.

### 1.1 Statement of need

Mature open codes for resistivity inversion exist: pyGIMLi (Rücker et al.
2017), SimPEG (Cockett et al. 2015) and ResIPy (Blanchy et al. 2020). Field
practice, including borehole siting in Nigeria, relies largely on RES2DINV and
IPI2Win. ertkit does not try to replace these. It fills a narrower gap: drilling
decisions are often made from resistivity images whose reliability is never
quantified. ertkit therefore treats three things as part of every result rather
than as optional extras:

1. a data error model and a misfit target that follows from it ($\chi^2 = N_d$, the number of data);
2. depth of investigation and sensitivity, used by default to mask the parts of
   a model that the data do not constrain;
3. parameter uncertainty and, for VES, the range of equivalent models.

It also serves as the computational engine for Kit 1 (VES inversion) of the
verification-first research kits project, and as the code behind the argument
of the blog post "Groundwater geophysics and inverse problems" (19 October 2026).

### 1.2 Why our own forward solvers

pyGIMLi and SimPEG are both installable with pip, so installation alone does not
justify new code. We write our own forward models and inversion because:

- **Transparency for teaching.** Kit 1 needs code that a student can read from
  end to end. Both forward models are small (a recurrence plus a Hankel
  transform for VES; P1 finite elements on a structured mesh for ERT).
- **Independent verification.** An independent implementation can be checked
  against pyGIMLi and SimPEG. A wrapper around them could not.
- **Light dependencies.** The core needs only numpy, scipy and libdlf (a
  BSD-licensed library of published digital filters that depends only on numpy).

pyGIMLi and SimPEG are used only for cross-checks, in an optional test job.

### 1.3 Users

- Researchers and graduate students in near-surface geophysics and hydrogeology.
- Nigerian supervisors and students, through Kit 1 (built on `ertkit.ves`).
- Practitioners who want a second, uncertainty-aware look at data already
  inverted elsewhere (from release 0.5, which reads RES2DINV files).

## 2. Scope by release

Each release must be complete, verified and reviewed before the next begins.

| Release | Content |
|---|---|
| 0.0 Foundation | Repository standards (section 6), docs site skeleton, release pipeline. No PyPI upload. |
| 0.1 VES | Layered-earth forward model, few-layer and smooth inversion, linearized uncertainty, equivalence analysis, CSV input. First PyPI release. |
| 0.2 ERT forward | 2.5-D finite-element forward model, standard arrays, pseudosections. |
| 0.3 ERT inversion | Gauss–Newton inversion with an error model and a $\chi^2$ target; adjoint sensitivities; optional blocky models. |
| 0.4 Reliability | Depth-of-investigation index, cumulative sensitivity, linearized uncertainty maps, masking by default. |
| 0.5 Field data | RES2DINV and pyGIMLi data formats, reciprocal-error analysis, one open field dataset end to end. |
| 1.0 | Interface frozen, complete documentation, submission to the Journal of Open Source Software (JOSS). |

**Not in 1.0:** 3-D surveys, surface topography (first candidate for 1.1),
time-lapse inversion, induced polarization, joint inversion, machine learning,
and petrophysical translation of resistivity into water content (Archie-type
relations), which needs its own careful treatment of the clay ambiguity.

## 3. Mathematics: VES (release 0.1)

### 3.1 Forward model

A layered earth has resistivities $\rho_1, \dots, \rho_N$ and thicknesses
$h_1, \dots, h_{N-1}$; layer $N$ is a half-space. The potential on the surface
at distance $r$ from a point current source $I$ is

$$
V(r) = \frac{I}{2\pi} \int_0^\infty T_1(\lambda) \thinspace J_0(\lambda r) \thinspace d\lambda,
$$

where the resistivity transform follows the recurrence (Pekeris 1940;
Koefoed 1979)

$$
T_N(\lambda) = \rho_N, \qquad T_i(\lambda) = \frac{T_{i+1}(\lambda) + \rho_i \tanh(\lambda h_i)}{1 + T_{i+1}(\lambda) \tanh(\lambda h_i)/\rho_i}, \qquad i = N-1, \dots, 1.
$$

For a homogeneous earth $T_1 = \rho$ and $V = I\rho/(2\pi r)$. The half-space
part is subtracted analytically,

$$
V(r) = \frac{I\rho_1}{2\pi r} + \frac{I}{2\pi} \int_0^\infty \bigl[T_1(\lambda) - \rho_1\bigr] J_0(\lambda r) \thinspace d\lambda,
$$

so that the integrand decays exponentially. The remaining integral is evaluated
with a published digital linear filter for $J_0$ (for example Key 2012, taken
from libdlf). Each filter is first checked on analytic transform pairs, in
particular the Lipschitz integral

$$
\int_0^\infty e^{-a\lambda} J_0(\lambda r) \thinspace d\lambda = \frac{1}{\sqrt{a^2 + r^2}},
$$

which is exactly the structure of the layered-earth kernel.

For a collinear array with current electrodes $A$, $B$ and potential electrodes
$M$, $N$, the apparent resistivity is

$$
\rho_a = K \frac{\Delta V}{I}, \qquad \Delta V = V(AM) - V(BM) - V(AN) + V(BN), \qquad K = \frac{2\pi}{1/AM - 1/BM - 1/AN + 1/BN}.
$$

This covers Schlumberger with finite $MN$, where $K = \pi(s^2 - b^2)/(2b)$
with $s = AB/2$ and $b = MN/2$, Wenner ($K = 2\pi a$), and any other
collinear array.

### 3.2 Inversion

Data are $d_i = \ln \rho_{a,i}$, with standard deviations from the error model
$\sigma(\rho_a) = \text{relative} \cdot \rho_a + \text{absolute}$
(LaBrecque et al. 1996), converted to log space as
$\sigma_i = \sigma(\rho_{a,i})/\rho_{a,i}$. The misfit is

$$
\chi^2 = \sum_{i=1}^{N_d} \left( \frac{d_i - f_i(m)}{\sigma_i} \right)^2,
$$

and a model that fits the data to the noise level has
$\chi^2 \approx N_d$, the number of data.

- **Few-layer inversion.** The unknowns are
  $m = (\ln \rho_i, \ln h_i)$. Minimize $\chi^2$ by Levenberg–Marquardt from
  several starting models (fixed seed), using the analytic derivative of the
  recurrence, checked against finite differences. Report the linearized
  covariance $(J^T W^2 J)^{-1}$, standard deviations as multiplicative
  factors, and the correlation matrix. A helper compares 2 to 6 layers using
  $\chi^2$ and the Bayesian information criterion.
- **Equivalence analysis.** A Metropolis sampler with fixed seed explores
  models with posterior proportional to $\exp(-\chi^2/2)$ within user bounds.
  The report gives the range of each $\rho_i$ and $h_i$, and of the Dar Zarrouk
  parameters $S_i = h_i/\rho_i$ and $T_i = h_i \rho_i$ (Maillet 1947), which
  expose the classic equivalence of thin conductive and resistive layers.
  Several chains are run and convergence is checked (acceptance rate,
  effective sample size, $\hat R$).
- **Smooth (Occam) inversion.** Many thin layers with fixed, logarithmically
  growing thicknesses and first-difference roughness on $\ln \rho$. At each
  iteration the regularization weight is chosen as the largest that still
  reduces $\chi^2$ toward its target $N_d$ (Constable et al. 1987).

### 3.3 Verification (acceptance criteria for 0.1)

1. Homogeneous half-space: $\rho_a = \rho$ to round-off for every array.
2. Two-layer earth against the image series below, to a tolerance justified by
   the filter's measured accuracy on analytic pairs.
3. Limits: $\rho_a \to \rho_1$ at small spacing and $\rho_a \to \rho_N$ at
   large spacing.
4. Jacobian: analytic derivative against finite differences.
5. Recovery: synthetic three- and four-layer models (including a weathered and
   fractured basement profile typical of Nigeria's basement complex) recovered
   within their reported uncertainty.
6. **Honest uncertainty.** Over many noise realizations, $\chi^2/N_d$ averages
   about 1, and the linearized 68% intervals contain the true parameters about
   68% of the time (within a stated statistical tolerance).
7. Equivalence: for a thin conductive layer, the sampler finds $S$ much better
   determined than $h$ and $\rho$ separately.
8. Cross-check: forward responses agree with pyGIMLi's VES modelling.

The two-layer image series, with reflection coefficient
$k = (\rho_2 - \rho_1)/(\rho_2 + \rho_1)$ and first-layer thickness $h$, is

$$
V(r) = \frac{I\rho_1}{2\pi r} \left[ 1 + 2 \sum_{n=1}^{\infty} \frac{k^n}{\sqrt{1 + (2nh/r)^2}} \right].
$$

## 4. Mathematics: 2.5-D ERT (releases 0.2 to 0.4)

### 4.1 Forward model

The conductivity $\sigma(x, z)$ varies along the profile ($x$) and with depth
($z$), but not along strike ($y$); electrodes are point sources on the surface.
The cosine transform in $y$,

$$
\tilde u(x, k, z) = \int_0^\infty u(x, y, z) \cos(ky) \thinspace dy, \qquad u(x, 0, z) = \frac{2}{\pi} \int_0^\infty \tilde u(x, k, z) \thinspace dk,
$$

turns the 3-D problem into a family of 2-D problems, one per wavenumber $k$:

$$
-\nabla \cdot (\sigma \nabla \tilde u) + k^2 \sigma \tilde u = \text{(source at the electrode)}.
$$

For a homogeneous half-space $\tilde u = (I\rho/2\pi) K_0(kr)$, which fixes the
source scaling and is the first verification target.

- **Boundary conditions.** There is no flux through the surface. On the sides
  and bottom the mixed condition of Dey and Morrison (1979) applies,
  $\partial \tilde u/\partial n + k \thinspace [K_1(kr)/K_0(kr)] \cos\theta \thinspace \tilde u = 0$,
  where $\theta$ is the angle between the source direction and the outward
  normal.
- **Wavenumber quadrature.** The inverse transform is approximated as
  $u \approx \sum_j w_j \tilde u(k_j)$, with Gauss–Legendre nodes on
  $[0, k_0]$ and Gauss–Laguerre nodes beyond, and $k_0$ scaled to the smallest
  electrode spacing (Kemna 2000), or the optimized sets of Xu et al. (2000).
  The choice is made by test: the quadrature applied to $K_0$ must reproduce
  $1/r$ to better than 0.1% over the survey's range of $r$.
- **Singularity removal.** Write $\tilde u = \tilde u_p + \tilde u_s$, with
  $\tilde u_p$ the analytic half-space solution for the conductivity
  $\sigma_0$ at the source. The secondary part solves the equation below,
  which has no point singularity (Lowry et al. 1989; Zhao and Yedlin 1996).
- **Discretization.** P1 finite elements on a structured, graded triangular
  mesh: electrodes on nodes, fine cells under the profile, geometric growth into
  padding several profile lengths wide and deep. Inversion parameters are the
  cells in the region of interest.
- **Efficiency.** One sparse LU factorization per wavenumber, reused for every
  electrode as a source; any four-electrode measurement follows by
  superposition.
- **Arrays.** Wenner, Schlumberger, dipole–dipole, pole–dipole, gradient and
  arbitrary ABMN tables; analytic geometric factors; pseudosections plotted at
  the median depths of investigation of Edwards (1977).

The secondary potential of the singularity removal satisfies

$$
-\nabla \cdot (\sigma \nabla \tilde u_s) + k^2 \sigma \tilde u_s = \nabla \cdot \bigl((\sigma - \sigma_0) \nabla \tilde u_p\bigr) - k^2 (\sigma - \sigma_0) \tilde u_p.
$$

### 4.2 Inversion

The unknowns are $m = \ln \rho$ on the parameter cells, and the data
$d = \ln \rho_a$ with the error model of section 3.2. Minimize

$$
\Phi(m) = \lVert W_d (d - f(m)) \rVert^2 + \lambda \lVert W_m (m - m_{\mathrm{ref}}) \rVert^2,
$$

with $W_m$ combining a small smallness term and anisotropic first differences
(separate horizontal and vertical weights). Gauss–Newton steps are computed
with preconditioned conjugate gradients and a backtracking line search;
$\lambda$ is reduced (cooling) until $\chi^2$ reaches $N_d$. The result object
records why the iteration stopped.

Sensitivities come from the adjoint (reciprocity) formula, for each
wavenumber

$$
\frac{\partial \tilde u_{AM}}{\partial \sigma_e} = -\int_{\Omega_e} \bigl( \nabla \tilde u_A \cdot \nabla \tilde u_M + k^2 \tilde u_A \tilde u_M \bigr) \thinspace d\Omega,
$$

integrated with the same quadrature and chained to $\ln \rho_a$ and
$\ln \rho$; they are checked against finite differences. Optional L1 measures
of misfit (robust to outliers) and of model gradients (blocky models) use
iteratively reweighted least squares (Farquharson and Oldenburg 1998).

### 4.3 Reliability (0.4)

- **Depth-of-investigation index** (Oldenburg and Li 1999). Two inversions
  with different reference models give
  $R = (m_1 - m_2)/(m_{\mathrm{ref},1} - m_{\mathrm{ref},2})$; cells with
  $R$ above a threshold (default 0.1) are masked in plots unless the user
  turns masking off.
- **Cumulative sensitivity**, weighted by data errors and normalized by cell
  area.
- **Linearized uncertainty**: the diagonal of the posterior covariance,
  computed directly for models of a few thousand cells.

### 4.4 Verification (acceptance criteria for 0.2 to 0.4)

1. Quadrature reproduces $1/r$ from $K_0$ to better than 0.1%.
2. Homogeneous half-space: $\rho_a = \rho$ within 0.5% for every array.
3. Layered earth: 2.5-D results agree with the VES module of release 0.1.
4. Vertical contact: agreement with the analytic image solution
   (Telford et al. 1990).
5. Mesh refinement: errors decrease at the expected rate.
6. Reciprocity: swapping current and potential pairs leaves the transfer
   resistance unchanged to solver precision.
7. Sensitivities: adjoint against finite differences.
8. Recovery tests with fixed seeds: a conductive aquifer layer; a clay lens
   (conductive but not water-bearing, to show the ambiguity honestly); a
   weathered and fractured basement with a vertical fracture zone; a resistive
   sand aquifer under clay.
9. Cross-check: block models against pyGIMLi and SimPEG.

## 5. Software design

### 5.1 Package layout

```
src/ertkit/
    __init__.py        version and top-level names
    errors.py          ErrorModel
    results.py         InversionResult, shared by VES and ERT
    arrays.py          electrode configurations and geometric factors
    synthetic.py       benchmark models
    ves/               model.py, forward.py, invert.py, equivalence.py, io.py
    ert/               mesh.py, quadrature.py, forward.py, sensitivity.py,
                       invert.py, doi.py, io.py          (from 0.2)
    plot/              optional; needs the [plot] extra
```

### 5.2 Principles

- Small immutable dataclasses for inputs (models, surveys, error models) and a
  result object for every inversion (model, predicted data, $\chi^2$ history, λ
  history, iterations, convergence flag and message, uncertainty).
- SI units throughout (Ω·m, m, A, V), stated in every docstring.
- Every random function takes a `seed` or numpy `Generator`.
- No printing: `logging` for progress, `warnings` for non-convergence.
- Inputs validated with clear error messages.
- Fully typed public interface, shipped with `py.typed`.
- Kit 1 compatibility: the VES kernel recurrence, Hankel transform and array
  geometry are separate functions, so a teaching kit can replace one with a
  student's implementation and test it against ertkit's.

### 5.3 Interface sketch (0.1)

```python
import numpy as np
import ertkit as ek
from ertkit import ves

survey = ves.Schlumberger(ab2=np.geomspace(1.5, 150.0, 24), mn2=0.5)
truth = ves.LayeredModel(resistivity=[150.0, 30.0, 800.0], thickness=[2.0, 15.0])
rhoa = ves.forward(truth, survey)

data = ves.Sounding.synthetic(truth, survey, error=ek.ErrorModel(relative=0.03), seed=1)
# or: data = ves.read_csv("site_12.csv")

fit = ves.invert_layers(data, n_layers=3, seed=0)
fit.model, fit.chi2, fit.converged, fit.std, fit.correlation

ens = ves.equivalence(data, fit, n_samples=20_000, seed=0)
ens.ranges()        # rho, h, S and T for each layer

smooth = ves.invert_smooth(data)    # Occam; stops at chi2 = N
```

### 5.4 Interface sketch (0.2 to 0.4)

```python
from ertkit import ert

survey = ert.Survey.dipole_dipole(n_electrodes=48, spacing=2.0, max_n=6)
mesh = ert.Mesh2D.for_survey(survey)
model = ek.synthetic.weathered_basement(mesh)
data = ert.simulate(model, survey, error=ek.ErrorModel(relative=0.03, absolute=1e-4), seed=1)

result = ert.invert(data, mesh)
doi = ert.doi_index(data, mesh)
```

## 6. Engineering standards

### 6.1 Packaging

- `src/` layout, hatchling with hatch-vcs (version from the git tag), MIT
  license.
- Core dependencies: numpy, scipy, libdlf. Extras: `plot` (matplotlib),
  `docs`, `test`, `crosscheck` (pyGIMLi, SimPEG).
- Minimum versions follow SPEC 0 and rise at each release. As of October 2026:
  Python ≥ 3.12, numpy ≥ 2.2, scipy ≥ 1.15, matplotlib ≥ 3.10.
- Each GitHub release publishes to PyPI by trusted publishing (no stored
  tokens) and to Zenodo by webhook.

### 6.2 Continuous integration

| Job | Content |
|---|---|
| Quality | ruff lint and format check; mypy in strict mode on `src/` |
| Tests | Linux, macOS and Windows × Python 3.12 to 3.15 (3.15 allowed to fail until numpy and scipy ship wheels for it) |
| Oldest dependencies | the test suite against the minimum supported versions |
| Coverage | at least 90% of lines in the core; summary in the job log |
| Docs | Sphinx build with the gallery executed and warnings treated as errors; link check |
| Verification | the verification suite; its report is published with the docs |
| Cross-check | pyGIMLi and SimPEG comparisons; weekly and before each release; not blocking |
| Publish | on release: build sdist and wheel, upload to PyPI |

### 6.3 Tests

Seven kinds, each in its own directory or marker: unit tests; analytic
verification; cross-module consistency (2.5-D against VES); regression tests
on stored outputs; recovery tests; property-based tests with Hypothesis for
invariants (reciprocity, positivity, limits, symmetry); and the optional
cross-check against other codes. Every tolerance has a written justification
next to it.

### 6.4 Documentation

- Website built with Sphinx (pydata theme, numpydoc, sphinx-gallery, MyST) and
  published at https://abdgafartunde.github.io/ertkit/: getting started, user
  guide (theory), example gallery, verification report, API reference,
  changelog, contributing, how to cite.
- A LaTeX technical note for citation and teaching, with its tables generated
  by the verification suite.
- **No hand-typed results.** Every number in the documentation, README or
  technical note is produced by code that CI runs.

### 6.5 Process

- One issue and one pull request per feature; each pull request carries its
  tests, documentation and changelog entry, and merges only with CI green.
- An independent two-agent review (mathematics and code; documentation and
  consistency) before every release.
- Semantic versioning. Before 1.0, interface changes are allowed with changelog
  notes; deprecated names warn for one minor release before removal.
- CHANGELOG (Keep a Changelog format), CONTRIBUTING (setup with uv, tests,
  release checklist), CODE_OF_CONDUCT (Contributor Covenant 2.1), issue and
  pull request templates, CITATION.cff, .zenodo.json.

## 7. Release 0.0 checklist

- [ ] `src/ertkit` skeleton with `__version__`, `py.typed` and LICENSE
- [ ] pyproject.toml with build, extras, ruff, mypy, pytest and coverage settings
- [ ] CI workflows: quality and tests, docs build and deploy, publish
- [ ] Docs site skeleton live at abdgafartunde.github.io/ertkit
- [ ] README rewritten: statement of need, status, roadmap, installation
- [ ] CHANGELOG, CONTRIBUTING, CODE_OF_CONDUCT, CITATION.cff, .zenodo.json,
      issue and pull request templates
- [ ] PyPI trusted publisher configured (needs your PyPI account)

## 8. Risks

| Risk | Mitigation |
|---|---|
| 2.5-D accuracy near electrodes | Singularity removal; quadrature and half-space gates in CI |
| Local minima in few-layer VES | Several starting models; equivalence sampling |
| Scope creep | Non-goals in section 2; each release complete before the next |
| Field data licensing | Use only data with a clear open license; Nigerian data only with written permission |
| Fragile installs of cross-check tools | Cross-check job is optional and does not block merges |
| New Python versions without wheels | Allowed to fail until wheels exist |
| Limited time (weekends) | Releases are defined by content and acceptance criteria, not dates |

## 9. Open questions

1. Can you or your contacts (for example at FUT Minna) share VES or ERT field
   data, with permission to publish? Nigerian data would strengthen 0.1, 0.5
   and Kit 1 considerably.
2. Is the documentation address abdgafartunde.github.io/ertkit acceptable?
3. Should the technical note be written for 0.1 already, or from 0.3 onward?

## References

- Blanchy, G., Saneiyan, S., Boyd, J., McLachlan, P., Binley, A. (2020).
  ResIPy, an intuitive open source software for complex geoelectrical
  inversion/modeling. *Computers & Geosciences*, 137, 104423.
- Cockett, R., Kang, S., Heagy, L. J., Pidlisecky, A., Oldenburg, D. W. (2015).
  SimPEG: An open source framework for simulation and gradient based parameter
  estimation in geophysical applications. *Computers & Geosciences*, 85, 142–154.
- Constable, S. C., Parker, R. L., Constable, C. G. (1987). Occam's inversion:
  a practical algorithm for generating smooth models from electromagnetic
  sounding data. *Geophysics*, 52(3), 289–300.
- Dey, A., Morrison, H. F. (1979). Resistivity modelling for arbitrarily shaped
  two-dimensional structures. *Geophysical Prospecting*, 27(1), 106–136.
- Edwards, L. S. (1977). A modified pseudosection for resistivity and IP.
  *Geophysics*, 42(5), 1020–1036.
- Farquharson, C. G., Oldenburg, D. W. (1998). Non-linear inversion using
  general measures of data misfit and model structure. *Geophysical Journal
  International*, 134(1), 213–227.
- Kemna, A. (2000). *Tomographic inversion of complex resistivity: theory and
  application*. PhD thesis, Ruhr-Universität Bochum.
- Key, K. (2012). Is the fast Hankel transform faster than quadrature?
  *Geophysics*, 77(3), F21–F30.
- Koefoed, O. (1979). *Geosounding Principles 1: Resistivity Sounding
  Measurements*. Elsevier.
- LaBrecque, D. J., Miletto, M., Daily, W., Ramirez, A., Owen, E. (1996). The
  effects of noise on Occam's inversion of resistivity tomography data.
  *Geophysics*, 61(2), 538–548.
- Lowry, T., Allen, M. B., Shive, P. N. (1989). Singularity removal: a
  refinement of resistivity modeling techniques. *Geophysics*, 54(6), 766–774.
- Maillet, R. (1947). The fundamental equations of electrical prospecting.
  *Geophysics*, 12(4), 529–556.
- Oldenburg, D. W., Li, Y. (1999). Estimating depth of investigation in DC
  resistivity and IP surveys. *Geophysics*, 64(2), 403–416.
- Pekeris, C. L. (1940). Direct method of interpretation in resistivity
  prospecting. *Geophysics*, 5(1), 31–42.
- Rücker, C., Günther, T., Wagner, F. M. (2017). pyGIMLi: An open-source library
  for modelling and inversion in geophysics. *Computers & Geosciences*, 109,
  106–123.
- Telford, W. M., Geldart, L. P., Sheriff, R. E. (1990). *Applied Geophysics*,
  2nd ed. Cambridge University Press.
- Xu, S., Duan, B., Zhang, D. (2000). Selection of the wavenumbers k using an
  optimization method for the inverse Fourier transform in 2.5D electrical
  modelling. *Geophysical Prospecting*, 48(5), 789–796.
- Zhao, S., Yedlin, M. J. (1996). Some refinements on the finite-difference
  method for 3-D dc resistivity modeling. *Geophysics*, 61(5), 1301–1307.
