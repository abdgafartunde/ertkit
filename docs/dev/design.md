# ertkit design document

Status: draft, revised after review of the mathematics and of the software
design, 14 October 2026. Nothing here is implemented yet. Names in the
code sketches may change during implementation; the public API becomes stable
at 1.0.

## 1. Purpose

ertkit is a Python package for inverting vertical electrical sounding (VES)
and 2.5-D electrical resistivity tomography (ERT) data in groundwater
exploration. Its distinguishing aim is that each result states what the data
can and cannot resolve:

- VES results from release 0.1: linearized uncertainty, sensitivity and a 1-D
  depth of investigation; equivalence analysis from 0.2.
- ERT results from release 0.4: cumulative sensitivity; and from 0.5: the
  depth-of-investigation index and uncertainty maps, used by default to mask
  unresolved regions.

### 1.1 Statement of need

Drilling decisions in groundwater exploration are often made from resistivity
models whose reliability is never quantified. Practitioners, including those
siting boreholes in Nigeria, rely largely on RES2DINV and IPI2Win, which report
a best-fitting model. ertkit treats three things as part of every result rather
than as optional extras:

1. a data error model and a misfit target that follows from it
   ($\chi^2 \approx N_d$, the number of data);
2. sensitivity and depth of investigation, used to mask what the data do not
   constrain;
3. parameter uncertainty and, for VES, the range of equivalent models.

This puts into practice the argument of the blog post "Groundwater geophysics
and inverse problems" (19 October 2026).

### 1.2 Build or contribute

Mature open codes exist:

- pyGIMLi (Rücker et al. 2017; Apache-2.0): VES and 2.5-D ERT modeling and
  inversion, with depth-of-investigation tools;
- SimPEG (Cockett et al. 2015; MIT): 1-D and 2.5-D DC resistivity, with
  target-misfit directives;
- ResIPy (Blanchy et al. 2020; GPL-3.0): a front end to the R2 and R3t codes,
  with error models and depth of investigation.

For 2.5-D ERT alone, these tools are sufficient, and ertkit's ERT part
reimplements established methods. ertkit is built anyway for four reasons:

1. **VES reliability by default.** Linearized uncertainty, sensitivity and
   equivalence analysis are the standard output of a VES inversion, not an
   extra step.
2. **Verification as a published artifact.** A verification report, generated
   by continuous integration on every change, documents each check and its
   tolerance.
3. **Transparency for teaching.** The VES engine is small enough to read end to
   end, which a teaching kit needs (section 1.3).
4. **Independence.** An independent implementation can be cross-checked
   against pyGIMLi and SimPEG; consistent reliability outputs across VES and ERT
   require owning both.

Where cross-checks reveal problems in other codes, they are reported upstream.

**Provenance.** No code is adapted from ResIPy (GPL-3.0). Code adapted from
pyGIMLi (Apache-2.0) carries attribution. Filter coefficients come from libdlf
(code BSD-3-Clause; filter values CC BY 4.0, attributed in the documentation).

### 1.3 Users

- Researchers and graduate students in near-surface geophysics and
  hydrogeology.
- Teaching: Kit 1 (VES inversion) of a planned set of verification-first
  teaching kits for Nigerian STEM students. Kit 1 is planned separately and
  will build on `ertkit.ves`; its requirements (for example, running offline
  and on modest laptops) will be set when it is planned.
- Practitioners who want a second, uncertainty-aware look at data already
  inverted elsewhere (from release 0.6, which reads RES2DINV files).

## 2. Scope by release

| Release | Content |
|---|---|
| 0.0 Foundation | A milestone, not a tag: repository standards (section 6), documentation site, release pipeline rehearsed on TestPyPI. |
| 0.1 VES core | Layered-earth forward model; few-layer inversion with linearized uncertainty; smooth (Occam) inversion with sensitivity and 1-D depth of investigation; CSV input. First PyPI release. |
| 0.2 VES equivalence | Equivalence analysis by Markov chain Monte Carlo with convergence diagnostics; comparison of the number of layers. |
| 0.3 ERT forward | 2.5-D finite-element forward model, standard arrays, pseudosections. |
| 0.4 ERT inversion | Gauss–Newton inversion with an error model and a $\chi^2$ target; adjoint sensitivities; cumulative sensitivity; optional robust and blocky (L1) measures. |
| 0.5 ERT reliability | Depth-of-investigation index, linearized uncertainty maps, masking by default. |
| 0.6 Field data | RES2DINV and pyGIMLi data formats, reciprocal-error analysis, one open field dataset end to end. |
| 1.0 | Public API declared stable; documentation covers every public function and the theory of each release. |

Patch releases (0.x.y) may ship while the next minor release is in progress.

**JOSS.** A submission to the Journal of Open Source Software follows once its
screening criteria hold: at least six months of public history with releases,
issues and pull requests; at least one research output or documented adoption
(for example a Kit 1 trial or a preprint using ertkit); and the AI-use record
of section 6.5.

**Not in 1.0:** 3-D surveys; surface topography (first candidate after 1.0;
until then, readers warn about or reject files with topography); time-lapse
inversion; induced polarization; joint inversion; machine learning; and
petrophysical translation of resistivity into water content (Archie-type
relations), which needs its own careful treatment of the clay ambiguity.

## 3. Mathematics: VES (releases 0.1 and 0.2)

**Notation.** $\rho$ is resistivity and $\sigma = 1/\rho$ conductivity;
$\mathcal{T}$ is the resistivity transform; $\lambda$ is the Hankel-transform
variable; $\kappa$ is a reflection coefficient; $K$ is the geometric factor
(distinct from the Bessel functions $K_0$ and $K_1$ of section 4); $s$ denotes
data standard deviations; $\mu$ is a regularization weight; $N_d$ is the
number of data and $p$ the number of free parameters.

### 3.1 Forward model

A layered earth has resistivities $\rho_1, \dots, \rho_N$ and thicknesses
$h_1, \dots, h_{N-1}$; layer $N$ is a half-space. The potential on the surface
at distance $r$ from a point current source $I$ is

$$
V(r) = \frac{I}{2\pi} \int_0^\infty \mathcal{T}_1(\lambda) \thinspace J_0(\lambda r) \thinspace d\lambda,
$$

where the resistivity transform follows the recurrence (Pekeris 1940;
Koefoed 1979)

$$
\mathcal{T}_N(\lambda) = \rho_N, \qquad \mathcal{T}_i(\lambda) = \frac{\mathcal{T}_{i+1}(\lambda) + \rho_i \tanh(\lambda h_i)}{1 + \mathcal{T}_{i+1}(\lambda) \tanh(\lambda h_i)/\rho_i}, \qquad i = N-1, \dots, 1.
$$

For a homogeneous earth $\mathcal{T}_1 = \rho$ and $V = I\rho/(2\pi r)$. The
integral is evaluated with a digital linear filter for $J_0$ (Ghosh 1971).
The default is the 120-point filter of Guptasarma and Singh (1997), taken from
libdlf, because it is accurate across the resistivity contrasts met in
practice: filters designed for controlled-source electromagnetics, such as
Key (2012), lose accuracy at high contrast. Subtracting the half-space part,

$$
V(r) = \frac{I\rho_1}{2\pi r} + \frac{I}{2\pi} \int_0^\infty \bigl[\mathcal{T}_1(\lambda) - \rho_1\bigr] J_0(\lambda r) \thinspace d\lambda,
$$

is used only if it does not worsen the accuracy gate of section 3.5; it
amplifies filter errors when a resistive layer lies over a conductive one.

For a collinear array with current electrodes $A$, $B$ and potential electrodes
$M$, $N$, the apparent resistivity is

$$
\rho_a = K \frac{\Delta V}{I}, \qquad \Delta V = V(AM) - V(BM) - V(AN) + V(BN), \qquad K = \frac{2\pi}{1/AM - 1/BM - 1/AN + 1/BN}.
$$

This covers Schlumberger soundings with finite $MN$, where
$K = \pi(s_{AB}^2 - b^2)/(2b)$ with $s_{AB} = AB/2$ and $b = MN/2$, Wenner
soundings ($K = 2\pi a$), and any other collinear array. Field Schlumberger
soundings change $MN$ partway, with overlapping readings: $AB/2$ and $MN/2$
are stored for every reading, and an optional static-shift factor per $MN$
segment can be estimated with the model.

### 3.2 Error model and misfit

Measurement errors are specified on the transfer resistance $R = \Delta V/I$
(LaBrecque et al. 1996):

$$
s_R = a + b \thinspace |R|,
$$

with $a$ in ohms and $b$ relative. Since $\rho_a = KR$, the data
$d_i = \ln \rho_{a,i}$ have log-space standard deviations
$s_i = b + a/|R_i|$. Unless reciprocal measurements justify less, $b$ includes
a modeling-error floor of 2 to 3% (Tso et al. 2017), because measurement noise
alone leads to overfitting field data. Data with $R_i \le 0$ or
$|R_i| < 3a$ are removed with a warning, since their logarithm is undefined or
dominated by noise. The misfit is

$$
\chi^2 = \sum_{i=1}^{N_d} \left( \frac{d_i - f_i(m)}{s_i} \right)^2.
$$

A fit with $p$ free parameters that matches the data to the noise level has
$\chi^2 \approx N_d - p$; a regularized (smooth) inversion targets
$\chi^2 = N_d$.

### 3.3 Inversion (0.1)

- **Few-layer inversion.** The unknowns are $m = (\ln \rho_i, \ln h_i)$,
  with $p = 2N - 1$ parameters plus any static-shift factors. Minimize
  $\chi^2$ by Levenberg–Marquardt from several starting models. The Jacobian
  is computed by complex-step differentiation of the recurrence and the
  filter, which is accurate to round-off. Report the linearized covariance
  $(J^T W^2 J)^{-1}$, where $W = \mathrm{diag}(1/s_i)$; the standard deviation
  of each log parameter, the corresponding multiplicative factor (for example
  $\rho_i$ known within a factor of 1.2), the correlation matrix, and the
  depths $z_i = h_1 + \dots + h_i$ with their standard deviations, since
  drilling decisions use depth. A goodness-of-fit $p$-value with $N_d - p$
  degrees of freedom is reported, with a warning below 0.01.
- **Smooth (Occam) inversion.** Many thin layers with fixed, logarithmically
  growing thicknesses and first-difference roughness $\partial$ on
  $\ln \rho$. Following Constable et al. (1987), each iteration solves for the
  model itself,
  $m_{k+1}(\mu) = \bigl[\mu \partial^T \partial + (WJ)^T WJ\bigr]^{-1} (WJ)^T W \hat d$
  with $\hat d = d - f(m_k) + J m_k$. While the target $\chi^2 = N_d$ cannot
  be reached, $\mu$ minimizes the nonlinear $\chi^2$; once it can, $\mu$ is
  the largest value with $\chi^2 = N_d$. The result includes the sensitivity
  of the data to each layer and a 1-D depth of investigation.

### 3.4 Equivalence and number of layers (0.2)

- **Equivalence analysis.** The posterior is proportional to
  $\exp(-\chi^2/2)$, with a uniform prior on $(\ln \rho_i, \ln h_i)$ within
  user-given bounds. Along an equivalence ridge the likelihood is flat, so
  individual $\rho_i$ and $h_i$ are limited only by these bounds. The sampler
  is Metropolis with adaptive proposals during warm-up (Haario et al. 2001),
  or an ensemble sampler, with several chains. The report gives 5% to 95%
  intervals of each $\rho_i$, $h_i$ and depth, and of the Dar Zarrouk
  parameters $S_i = h_i/\rho_i$ and $T_i = h_i \rho_i$ (Maillet 1947); an
  interval that reaches a bound is flagged as limited by the prior. Results
  count as converged only if the rank-normalized split $\hat R < 1.01$ and the
  bulk and tail effective sample sizes are at least 400 (Vehtari et al. 2021);
  otherwise the report says "not converged". The sampler is validated by
  simulation-based calibration (Talts et al. 2018).
- **Number of layers.** A helper compares fits with 2 to 6 layers using the
  goodness-of-fit $p$-value and the Bayesian information criterion.

### 3.5 Verification (acceptance criteria for 0.1 and 0.2)

Every tolerance below is checked in a test and reported in the verification
report.

1. **Filter accuracy.** On two-layer earths with resistivity contrasts from
   $10^{-3}$ to $10^{3}$ and $r/h$ from 0.1 to 100, the error in $V$ and in
   Schlumberger $\rho_a$ with finite $MN$, against the image series below, is at
   most $10^{-6}$ relative. The series is summed until $|\kappa|^n < 10^{-14}$,
   with the contrast capped so that this stays affordable.
2. **Half-space.** Computed without subtraction, so that the filter weights
   are tested: $\rho_a = \rho$ to $10^{-6}$ relative for every array.
3. **Limits.** $\rho_a$ approaches $\rho_1$ at small spacing and $\rho_N$ at
   large spacing, with the deviation at the end spacings matching the image
   series.
4. **Jacobian.** Complex-step derivatives against central finite differences,
   relative error below $10^{-6}$.
5. **Recovery.** Synthetic three- and four-layer models, including a
   weathered and fractured basement profile typical of Nigeria's basement
   complex, are recovered with every parameter within 3 standard deviations
   and a Mahalanobis distance below the 99.7% quantile of $\chi^2_p$.
6. **Honest uncertainty.** For a fixed, well-resolved model (all linearized
   standard deviations of log parameters below 0.2), with the true number of
   layers and 1000 noise draws from the assumed error model: the mean of the
   minimum $\chi^2$ lies within $(N_d - p) \pm 3\sqrt{2(N_d - p)/1000}$; the
   fraction of estimates within one standard deviation of the truth lies within
   $0.683 \pm 0.044$ for each parameter; and the fraction with Mahalanobis
   distance within the 68.3% quantile of $\chi^2_p$ lies in the same band. A
   model affected by equivalence reports its coverage without a pass or fail
   threshold, to show why the sampler is needed. No coverage claim is made for
   smooth inversions.
7. **Occam.** The final $\chi^2$ is within $N_d \pm \sqrt{2N_d}$, and raising
   $\mu$ by 10% takes $\chi^2$ above $N_d$.
8. **Equivalence (0.2).** For a thin conductive layer, the 90% interval of
   $\ln S$ is at most 0.2 times the widths for $\ln h$ and $\ln \rho$, and their
   posterior correlation is at least 0.95; likewise for $T$ on a thin resistive
   layer.
9. **Cross-check.** Forward responses agree with pyGIMLi's VES modeling within
   0.1%, and with SimPEG run with a different filter.

The two-layer image series, with reflection coefficient
$\kappa = (\rho_2 - \rho_1)/(\rho_2 + \rho_1)$ and first-layer thickness $h$, is

$$
V(r) = \frac{I\rho_1}{2\pi r} \left[ 1 + 2 \sum_{n=1}^{\infty} \frac{\kappa^n}{\sqrt{1 + (2nh/r)^2}} \right].
$$

### 3.6 Documented limitations of VES

The documentation states, with examples: the paradox of anisotropy (only
$\sqrt{\rho_t \rho_l}$ and the thickness scaled by the coefficient of
anisotropy can be resolved, so depths come out too large); suppression of thin
layers; the classic curve types (H, K, A, Q), for teaching; the effect of
electrode position errors (Zhou and Dahlin 2003) and of remote-electrode
positions; and the assumption of a flat, laterally uniform earth (see Binley
and Slater 2020 for these topics in depth).

## 4. Mathematics: 2.5-D ERT (releases 0.3 to 0.5)

### 4.1 Forward model

The conductivity $\sigma(x, z)$ varies along the profile ($x$) and with depth
($z$), but not along strike ($y$); electrodes are point sources on a flat
surface. The cosine transform in $y$,

$$
\tilde u(x, k, z) = \int_0^\infty u(x, y, z) \cos(ky) \thinspace dy, \qquad u(x, 0, z) = \frac{2}{\pi} \int_0^\infty \tilde u(x, k, z) \thinspace dk,
$$

turns the 3-D problem into a family of 2-D problems on the half-plane, one per
wavenumber $k$:

$$
-\nabla \cdot (\sigma \nabla \tilde u) + k^2 \sigma \tilde u = \frac{I}{2} \thinspace \delta(x - x_s) \thinspace \delta(z).
$$

For a homogeneous half-space $\tilde u = (I\rho/2\pi) K_0(kr)$.

- **Boundary conditions.** There is no flux through the surface. On the sides
  and bottom, the mixed condition of Dey and Morrison (1979),
  $\partial \tilde u/\partial n + \alpha \tilde u = 0$, uses one fixed
  reference point $x_c$ (the center of the electrode spread, at the surface)
  for all sources, as pyGIMLi and SimPEG do, so that one factorization serves
  every electrode and reciprocity holds exactly:
  $\alpha = k \thinspace [K_1(kr)/K_0(kr)] \cos\theta$, with $r = |x - x_c|$
  and $\cos\theta = (x - x_c) \cdot n/r$, so $\alpha \ge 0$ on a convex
  domain. The ratio is computed with exponentially scaled Bessel functions
  (`scipy.special.k1e` and `k0e`), which do not overflow at large $kr$. The
  padding is chosen so that doubling it changes every $\rho_a$ by less than
  $10^{-4}$.
- **Wavenumber quadrature.** Gauss–Legendre nodes on $[0, k_0]$ with the
  substitution $k = k_0 t^2$ (which removes the logarithmic singularity of
  $K_0$ at $k = 0$), and Gauss–Laguerre nodes beyond with $k = k_0(1 + t)$,
  where $k_0 = 1/(2 r_{\min})$ (Kemna 2000, as implemented in pyGIMLi; SimPEG
  instead fits optimized wavenumbers in the manner of Xu et al. 2000). The
  quadrature is applied only to the
  secondary field (below), so the total potential is
  $u = I\rho_0/(2\pi r) + \sum_j w_j \tilde u_s(k_j)$, with $w_j$ including
  the factor $2/\pi$.
- **Singularity removal.** Write $\tilde u = \tilde u_p + \tilde u_s$, with
  $\tilde u_p = (I/2\pi\sigma_0) K_0(kr)$ the analytic half-space solution for
  the reference conductivity $\sigma_0$ (Lowry et al. 1989; Zhao and Yedlin
  1996). With the discrete operator $A(\sigma) = S(\sigma) + k^2 M(\sigma) + B(\sigma)$
  (stiffness, consistent mass and boundary terms), the secondary field solves
  the equation below, which includes every boundary term automatically.
  $\sigma_0$ is the conductivity of the cells next to the source electrode
  (area-weighted, with a warning, if they differ); during an inversion it is
  frozen within each iteration and in finite-difference checks.
- **Discretization.** P1 finite elements on a structured, graded triangular
  mesh with mirrored diagonals (which keeps the mesh symmetric), electrodes on
  nodes, and geometric growth into the padding. The assembly works for any
  triangulation, so that topography later needs only a different mesh.
- **Region of interest.** Inversion parameters cover the electrode spread
  laterally and reach 1.5 to 2 times the largest median depth of
  investigation; fine mesh cells are grouped into parameter cells, and padding
  cells take the value of the nearest parameter cell.
- **Arrays.** Wenner, Schlumberger, dipole–dipole, pole–dipole, pole–pole,
  gradient and arbitrary ABMN tables; analytic geometric factors;
  pseudosections plotted at the median depths of investigation of Edwards
  (1977), computed numerically for general arrays (Barker 1989).

The secondary field satisfies

$$
A(\sigma) \thinspace \tilde u_s = -\bigl[A(\sigma) - A(\sigma_0)\bigr] \thinspace \tilde u_p.
$$

### 4.2 Inversion (0.4)

The unknowns are $m = \ln \rho$ on the parameter cells, and the data
$d = \ln \rho_a$ with the error model of section 3.2. Minimize

$$
\Phi(m) = \lVert W (d - f(m)) \rVert^2 + \mu \lVert W_m (m - m_{\mathrm{ref}}) \rVert^2,
$$

with $W_m$ combining a smallness term and anisotropic first differences
weighted by the distance between cell centers (so that the regularization
does not depend on the mesh grading). For models of a few thousand cells, each
Gauss–Newton step is a direct Cholesky solve, followed by a backtracking line
search. The weight $\mu$ is reduced (cooling) until $\chi^2 \le 1.05 \thinspace N_d$
or until $\chi^2$ stops decreasing; the result records the reason.

Sensitivities come from the adjoint (reciprocity) formula. With $\tilde u_A$
and $\tilde u_M$ computed for sources $(I/2)\delta$ at $A$ and $M$, for each
wavenumber

$$
\frac{\partial \tilde u_A(M)}{\partial \sigma_e} = -\frac{2}{I} \int_{\Omega_e} \bigl( \nabla \tilde u_A \cdot \nabla \tilde u_M + k^2 \tilde u_A \tilde u_M \bigr) \thinspace d\Omega,
$$

implemented exactly as the discrete derivative
$-(2/I) \thinspace \tilde u_M^T (S_e + k^2 M_e + B_e) \thinspace \tilde u_A$,
then summed with the quadrature weights and chained to $\ln \rho_a$ and
$\ln \rho$. Since $\rho_a$ is homogeneous of degree one in $\rho$, the
sensitivities satisfy $\sum_e \partial \ln \rho_a/\partial \ln \rho_e = 1$
over all cells, including padding; this checks every factor at once.

Every ERT result includes the cumulative sensitivity
$S_j = \lVert J_{:,j}/s \rVert / A_j$, the error-weighted column norm of the
Jacobian divided by the cell area. Optional L1 measures of misfit (robust to
outliers) and of model gradients (blocky models) use iteratively reweighted
least squares (Farquharson and Oldenburg 1998); with an L1 misfit the target
is the expected L1 norm, about $N_d \sqrt{2/\pi}$ for Gaussian errors.

### 4.3 Reliability (0.5)

- **Depth-of-investigation index** (Oldenburg and Li 1999). Two further
  inversions, with the same weights and the same $\chi^2$ target, use
  reference models of 0.1 and 10 times the mean resistivity (Marescot et al.
  2003). Then $R = (m_1 - m_2)/(m_{\mathrm{ref},1} - m_{\mathrm{ref},2})$.
  When $R$ at the bottom of the region of interest exceeds 0.2, $R$ is rescaled
  to reach 1 there; otherwise ramp reference models are used, as in the paper.
  Cells where the scaled $R$ exceeds 0.1 are masked in plots unless the user
  turns masking off.
- **Linearized uncertainty.** The diagonal of
  $(J^T W^2 J + \mu W_m^T W_m)^{-1}$, labeled as conditional on the
  regularization, since the regularization acts as a prior chosen from the
  data and the result has no frequentist coverage where the data are weak.
- **Resolution.** The diagonal of the model resolution matrix (Friedel 2003).

### 4.4 Verification (acceptance criteria for 0.3 to 0.5)

1. **Quadrature.** For every configuration in the survey, including pole–pole,
   the quadrature applied to the analytic half-space spectrum gives $\rho_a$
   within $10^{-4}$.
2. **Per wavenumber.** In total-field mode (no singularity removal), the
   finite-element solution matches $(I\rho/2\pi) K_0(kr)$ for a half-space and
   the two-layer $k$-domain series
   $(I\rho_1/2\pi)\bigl[K_0(kr) + 2\sum_{n \ge 1} \kappa^n K_0\bigl(k\sqrt{r^2 + (2nh)^2}\bigr)\bigr]$,
   within a tolerance taken from the measured mesh convergence.
3. **Layered earth.** 2.5-D apparent resistivities agree with the VES module
   within 1%.
4. **Vertical contact and dike.** Agreement with the analytic image-series
   solutions (for the contact, Telford et al. 1990) within 2%, with no
   electrode on a contact.
5. **Mesh refinement.** Uniform four-way refinement of a two-layer model with
   the interface on mesh lines, with the quadrature held fixed: the observed
   order from three levels lies in $[1.8, 2.2]$ for every configuration (as
   expected for P1 elements; Scott 1973).
6. **Reciprocity.** Swapping current and potential pairs changes the transfer
   resistance by less than $10^{-10}$ relative in total-field mode, and within
   the discretization tolerance with singularity removal.
7. **Sensitivities.** Adjoint against central finite differences (with
   $\sigma_0$ frozen), relative error below $10^{-5}$; and the sum rule of
   section 4.2 to $10^{-8}$.
8. **Median depths.** Computed median depths of investigation reproduce
   Edwards (1977), for example $z_e/a = 0.519$ for Wenner and 0.416 for
   dipole–dipole with $n = 1$.
9. **Recovery.** Data simulated on a finer, different mesh (no inverse crime),
   with stored noise vectors: a conductive aquifer layer; a weathered and
   fractured basement with a vertical fracture zone; a resistive sand aquifer
   under clay. Each benchmark states tolerances for the target's log
   resistivity, the interface depth, and the log-RMS error inside the
   depth-of-investigation mask. The clay ambiguity is shown separately, by two
   models with the same resistivity but different hydrogeological meaning.
10. **Reliability (0.5).** For a half-space, the cumulative sensitivity matches
    the analytic 3-D kernel integrated along strike. On a layered model the
    scaled depth-of-investigation index is below 0.1 near the surface and above
    0.9 at the bottom of the region of interest. A linear-Gaussian test (fixed
    Jacobian, models drawn from the prior implied by the regularization) checks
    the coverage of the computed uncertainties.
11. **Cross-check.** Block models and a dipping contact agree with pyGIMLi and
    SimPEG within 2%.

## 5. Software design

### 5.1 Package layout

```
src/ertkit/
    __init__.py        version, show_versions, top-level names
    noise.py           ErrorModel
    results.py         result base class and save/load
    arrays.py          electrode configurations and geometric factors
    benchmarks.py      mesh-free benchmark models
    _inversion.py      shared Gauss-Newton, Levenberg-Marquardt and Occam driver
    ves/               survey.py, model.py, forward.py, invert.py, equivalence.py, io.py
    ert/               mesh.py, quadrature.py, forward.py, sensitivity.py,
                       invert.py, doi.py, io.py          (from 0.3)
    plot/              optional; needs the [plot] extra
```

The shared inversion driver works with any forward model that provides
`predict(m)` and `jacobian(m)` (a typing `Protocol`), so VES and ERT use the
same tested code. The public API is what `__all__` lists and the API reference
documents. The building blocks a teaching kit replaces, `ves.resistivity_transform`,
`ves.hankel_j0` and `arrays.geometric_factor`, are public and become stable at
1.0.

### 5.2 Principles

- **Inputs** are dataclasses declared
  `@dataclass(frozen=True, eq=False, kw_only=True, slots=True)`. Their
  `__post_init__` converts arrays with `np.asarray(..., dtype=float)`,
  validates them, copies them and marks them read-only, so that frozen
  instances are really immutable.
- **Results** share a base class (model, predicted data, $\chi^2$ and its
  target, history, convergence flag, and a stop reason from a fixed set of
  strings), with subclasses for few-layer fits, smooth fits, ERT inversions and
  equivalence ensembles. Results can be saved and loaded.
- **Units** are SI (ohm meters, meters, amperes, volts), stated in every
  docstring.
- **Randomness** follows SPEC 7: functions take an `rng` argument (a seed or a
  numpy `Generator`), normalized with `np.random.default_rng`.
- **Messages.** There is no printing. Progress goes to
  `logging.getLogger(__name__)` (with a `NullHandler`), and non-convergence
  raises `ertkit.ConvergenceWarning`, a subclass of `UserWarning`.
- **Validation** of inputs gives clear error messages.
- **Typing.** The public interface is fully typed and ships `py.typed`; inputs
  are annotated as `ArrayLike` and outputs as `NDArray[np.float64]`.
- **Plotting is optional.** `import ertkit` never imports matplotlib;
  `ertkit.plot` raises an `ImportError` that names `pip install "ertkit[plot]"`.
- **Files** are always opened with an explicit encoding, because RES2DINV
  files are often not UTF-8 and the default encoding differs between Python
  3.12 to 3.14 and 3.15.
- **Bug reports** include the output of `ertkit.show_versions()`.

### 5.3 Interface sketch

```python
import numpy as np
import ertkit as ek
from ertkit import ert, ves

# VES (0.1 and 0.2)
survey = ves.Survey.schlumberger(ab2=np.geomspace(2.5, 150.0, 22), mn2=0.5)  # mn2 scalar or per reading
truth = ves.LayeredModel(resistivity=[150.0, 30.0, 800.0], thickness=[2.0, 15.0])
rhoa = ves.forward(truth, survey)                          # noise-free array, ohm m
noise = ek.ErrorModel(relative=0.03, absolute=1e-4)        # b and a of section 3.2; a in ohm
data = ves.Sounding.synthetic(truth, survey, noise, rng=1)  # or ves.read_csv(path)
fit = ves.invert_few_layer(data, n_layers=3, rng=0)        # LayerFit: .std_log, .factor, .correlation, .depths
smooth = ves.invert_smooth(data)                           # SmoothFit: .sensitivity, .doi_depth
ens = ves.sample_equivalence(data, fit, bounds=..., n_chains=4, n_samples=5_000, rng=0)

# ERT (0.3 to 0.5; provisional)
survey = ert.Survey.dipole_dipole(n_electrodes=48, spacing=2.0, max_n=6)
truth = ek.benchmarks.weathered_basement()                 # geometric, independent of any mesh
fine = ert.Mesh.for_survey(survey, refine=2)
data = ert.Dataset.synthetic(truth, survey, noise, mesh=fine, rng=1)
result = ert.invert(data, mesh=ert.Mesh.for_survey(survey))
doi = ert.doi_index(result)                                # reuses the result's settings
```

## 6. Engineering standards

### 6.1 Packaging and releases

- `src/` layout; MIT license declared with PEP 639 (`license = "MIT"`,
  `license-files = ["LICENSE"]`).
- Version from the git tag with hatchling and hatch-vcs:
  `local_scheme = "no-local-version"` (PyPI rejects local versions); the
  generated `_version.py` is ignored by git, ruff and coverage; CI checks out
  the full history (`fetch-depth: 0`); and `.git_archival.txt` (with
  `export-subst` in `.gitattributes`) lets the Zenodo source archive build
  without `.git`. Tags have the form `vX.Y.Z`.
- **Dependencies:** numpy, scipy and libdlf. The only published extra is
  `plot` (matplotlib). Development tools live in PEP 735 dependency groups
  (`test`, `docs`, `lint`, and `dev` combining them).
- **Minimum versions** are the lower of SPEC 0 and the versions in the oldest
  Google Colab runtime that can still be selected, so that ertkit installs in
  teaching notebooks without upgrading preinstalled packages. They are
  reviewed at each release and tested by the oldest-dependencies job. As of
  October 2026: Python ≥ 3.12, numpy ≥ 2.0, scipy ≥ 1.15, matplotlib ≥ 3.10.
  Python and numpy are below SPEC 0 on purpose: Colab currently runs Python
  3.13 with numpy 2.1.
- `uv.lock` is committed; the quality, docs and test jobs install with
  `uv sync --locked`; Dependabot updates the lock file and the workflow
  actions monthly, so CI meets new numpy and scipy versions before users do.
- **Release pipeline** (`release.yml`): every pull request builds the sdist and
  wheel and checks them (`twine check --strict`, `check-wheel-contents`, install
  and import). A manual run uploads a development version to TestPyPI; a
  published GitHub release uploads to PyPI by trusted publishing from the
  protected `pypi` environment, after checking that the tag matches the
  version in CITATION.cff and the top CHANGELOG heading. Zenodo is switched on
  just before 0.1; `.zenodo.json` carries no version, so it never goes stale.

### 6.2 Continuous integration

| Job | Content |
|---|---|
| Quality | pre-commit (ruff lint and format, file hygiene), mypy in strict mode on `src/`, from the locked environment |
| Tests | Ubuntu with Python 3.12, 3.13, 3.14 and 3.15; Windows with 3.12; macOS (arm64) with 3.15. Coverage is measured in the Ubuntu 3.14 job. |
| Oldest dependencies | Ubuntu, Python 3.12: `uv pip install --resolution lowest-direct -e ".[plot]" --group test`, then the tests not marked slow |
| Build | sdist and wheel built and checked, as in section 6.1 |
| Docs | Sphinx build with the gallery executed, `-W --keep-going`; deployed from `main` until 0.1, then on each release |
| Weekly (not blocking) | link check (with retries; publisher domains that block bots are ignored), and from 0.1 the cross-check job |

- **Cross-check job:** Ubuntu with Python 3.13, weekly and on manual dispatch,
  installing pinned versions from `ci/crosscheck.txt` (`pygimli>=1.6,<2`,
  `simpeg>=0.25,<0.26`), so their constraints stay out of `uv.lock`. Running it
  is part of the release checklist.
- **Workflow hygiene:** `permissions: contents: read` by default, `concurrency`
  with cancel-in-progress, `timeout-minutes` on every job, and
  `MPLBACKEND=Agg` and `PYTHONWARNDEFAULTENCODING=1` in the environment.
- New Python versions join CI when numpy and scipy publish wheels for them
  (usually at the first release candidate).

### 6.3 Tests

- **Markers** select what runs where: `slow` (excluded from the
  oldest-dependencies job), `verification` (feeds the verification report) and
  `crosscheck` (weekly job only).
- **Kinds of test:** unit tests; analytic verification; cross-module
  consistency (2.5-D against VES); regression tests on stored outputs;
  recovery tests; property-based tests with Hypothesis for cheap invariants
  (reciprocity, positivity, limits, symmetry); and cross-checks against other
  codes. Every tolerance has a written justification next to it.
- **Robust to platform differences.** Random streams and BLAS libraries differ
  between platforms, so regression tests compare with a relative tolerance
  derived from the solver's stopping tolerance; tests that depend on noise
  store the noise vector, not the seed; statistical tests assert intervals.
  Stored reference files change only through `pytest --regen` in a dedicated
  pull request.
- **Hypothesis** runs with a `ci` profile (`derandomize=True`,
  `deadline=None`, small `max_examples`).
- **Coverage:** branch coverage of `src/ertkit` excluding `plot` and
  `_version.py`, with `fail_under = 90`.
- pytest runs with `--strict-markers`, `--strict-config` and
  `filterwarnings = ["error"]`, so unexpected warnings fail the tests.

### 6.4 Documentation

- A website built with Sphinx (pydata theme, numpydoc, sphinx-gallery, MyST),
  published at https://abdgafartunde.github.io/ertkit/: getting started, user
  guide (theory), example gallery, verification report, API reference,
  changelog, contributing, how to cite. This design document lives in a
  developer section.
- **The theory pages are canonical.** A LaTeX technical note follows after
  release 0.4, built in CI from the same generated tables.
- **No hand-typed results.** Computed numbers come only from sphinx-gallery
  output; from the verification report, produced by
  `pytest -m verification`, whose plugin records each measured error with its
  tolerance in a JSON file that a small Sphinx extension renders (and that also
  writes the LaTeX tables); and from named constants (defaults, tolerances)
  documented through autodoc. Generated files are not committed. The README
  contains no computed numbers and links to the verification report.

### 6.5 Process

- One issue and one pull request per feature; each pull request carries its
  tests, documentation and changelog entry, and merges only when CI passes.
  A ruleset on `main` requires pull requests and passing checks and blocks force
  pushes, with no required approvals, so that a single author can merge.
- **Review before each release:** an AI-assisted review (mathematics and code;
  documentation and consistency), followed by the author's sign-off checklist
  (derivations, tolerances and the verification report checked personally).
  The outcome is recorded in the release notes.
- **AI use is recorded from release 0.0.** CONTRIBUTING states which AI tools
  are used and for what, and each release's notes summarize AI assistance and
  what the author verified. This also meets the JOSS disclosure policy.
- **Versions:** semantic versioning. Before 1.0, interface changes are allowed
  with changelog notes; after 1.0, breaking changes come only in a major
  release, after one minor release of deprecation warnings.
- **Project files:** CHANGELOG (Keep a Changelog 1.1.0), CONTRIBUTING
  (development setup with uv, tests, docs, release checklist, support
  expectations, AI use), CODE_OF_CONDUCT (Contributor Covenant 3.0), issue
  and pull request templates, CITATION.cff, `.zenodo.json`, `.mailmap`.

## 7. Release 0.0 (milestone)

### 7.1 Checklist

- [ ] README rewritten before 19 October (statement of need, status, roadmap,
      installation), since the website and a blog post link to it
- [ ] LICENSE at the repository root
- [ ] `src/ertkit` with `__init__.py`, `py.typed`, `show_versions`; a package
      test (version matches metadata, `py.typed` shipped, `import ertkit` does
      not import matplotlib)
- [ ] pyproject.toml and `uv.lock`
- [ ] `.gitignore`, `.gitattributes`, `.git_archival.txt`,
      `.pre-commit-config.yaml`, `.mailmap`
- [ ] Workflows: `ci.yml`, `docs.yml`, `release.yml`, `weekly.yml`; Dependabot
- [ ] Documentation site skeleton with one trivial gallery example
- [ ] CHANGELOG, CONTRIBUTING, CODE_OF_CONDUCT, CITATION.cff (no version or DOI
      until 0.1), issue and pull request templates
- [ ] Repository settings: Pages from GitHub Actions; ruleset on `main`;
      squash merges only; branches deleted after merge; wiki off; topics; the
      documentation as homepage
- [ ] PyPI and TestPyPI accounts with pending trusted publishers (owner
      `abdgafartunde`, repository `ertkit`, workflow `release.yml`,
      environments `pypi` and `testpypi`)

### 7.2 Acceptance criteria

1. All CI jobs pass and are required checks.
2. A development version uploaded to TestPyPI through trusted publishing
   installs and imports with the version derived from git.
3. The documentation site is live and builds with warnings treated as errors.
4. `sp-repo-review` passes, or each exception is justified in CONTRIBUTING.
5. CITATION.cff validates, and GitHub detects the license.

## 8. Risks

| Risk | Mitigation |
|---|---|
| 2.5-D accuracy near electrodes | Singularity removal; quadrature and half-space gates in CI |
| Local minima in few-layer VES | Several starting models; equivalence sampling (0.2) |
| Equivalence sampler fails to mix | Adaptive proposals; convergence diagnostics; "not converged" reported rather than hidden |
| Overconfident depth of investigation | The Oldenburg and Li procedure followed exactly, including rescaling and the choice of reference models |
| Scope creep | Non-goals in section 2; small releases with stated acceptance criteria |
| Field data licensing | Only data with a clear open license; Nigerian data only with written permission |
| Cross-check tools hard to install | Separate weekly job with pinned versions; not blocking |
| Platform-dependent numerics | Tolerance-based regression tests; stored noise vectors |
| Teaching installs break preinstalled packages | Minimum versions follow Colab as well as SPEC 0 |
| Limited time | Releases defined by content and acceptance criteria |

## 9. Open questions

1. **Field data.** VES or ERT data from Nigerian sites, shared with permission
   to publish, would strengthen releases 0.1 and 0.6 and Kit 1. They are to be
   requested from collaborators.
2. **Kit 1 requirements** (offline use, run time on modest laptops, an Octave
   version) are set when Kit 1 is planned; ertkit's examples should already run
   in under two minutes on a 4 GB laptop.

## References

- Barker, R. D. (1989). Depth of investigation of collinear symmetrical
  four-electrode arrays. *Geophysics*, 54(8), 1031–1037.
- Binley, A., Slater, L. (2020). *Resistivity and Induced Polarization:
  Theory and Applications to the Near-Surface Earth*. Cambridge University
  Press.
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
- Fox, R. C., Hohmann, G. W., Killpack, T. J., Rijo, L. (1980). Topographic
  effects in resistivity and induced-polarization surveys. *Geophysics*,
  45(1), 75–93.
- Friedel, S. (2003). Resolution, stability and efficiency of resistivity
  tomography estimated from a generalized inverse approach. *Geophysical
  Journal International*, 153(2), 305–316.
- Ghosh, D. P. (1971). The application of linear filter theory to the direct
  interpretation of geoelectrical resistivity sounding measurements.
  *Geophysical Prospecting*, 19(2), 192–217.
- Guptasarma, D., Singh, B. (1997). New digital linear filters for Hankel J0
  and J1 transforms. *Geophysical Prospecting*, 45(5), 745–762.
- Haario, H., Saksman, E., Tamminen, J. (2001). An adaptive Metropolis
  algorithm. *Bernoulli*, 7(2), 223–242.
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
- Marescot, L., Loke, M. H., Chapellier, D., Delaloye, R., Lambiel, C.,
  Reynard, E. (2003). Assessing reliability of 2D resistivity imaging in
  mountain permafrost studies using the depth of investigation index method.
  *Near Surface Geophysics*, 1(2), 57–67.
- Oldenburg, D. W., Li, Y. (1999). Estimating depth of investigation in DC
  resistivity and IP surveys. *Geophysics*, 64(2), 403–416.
- Pekeris, C. L. (1940). Direct method of interpretation in resistivity
  prospecting. *Geophysics*, 5(1), 31–42.
- Rücker, C., Günther, T., Wagner, F. M. (2017). pyGIMLi: An open-source library
  for modelling and inversion in geophysics. *Computers & Geosciences*, 109,
  106–123.
- Scott, R. (1973). Finite element convergence for singular data.
  *Numerische Mathematik*, 21(4), 317–327.
- Talts, S., Betancourt, M., Simpson, D., Vehtari, A., Gelman, A. (2018).
  Validating Bayesian inference algorithms with simulation-based calibration.
  arXiv:1804.06788.
- Telford, W. M., Geldart, L. P., Sheriff, R. E. (1990). *Applied Geophysics*,
  2nd ed. Cambridge University Press.
- Tso, C.-H. M., Kuras, O., Wilkinson, P. B., Uhlemann, S., Chambers, J. E.,
  Meldrum, P. I., Graham, J., Sherlock, E. F., Binley, A. (2017). Improved
  characterisation and modelling of measurement errors in electrical
  resistivity tomography (ERT) surveys. *Journal of Applied Geophysics*, 146,
  103–119.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., Bürkner, P.-C. (2021).
  Rank-normalization, folding, and localization: an improved R-hat for
  assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718.
- Xu, S., Duan, B., Zhang, D. (2000). Selection of the wavenumbers k using an
  optimization method for the inverse Fourier transform in 2.5D electrical
  modelling. *Geophysical Prospecting*, 48(5), 789–796.
- Zhao, S., Yedlin, M. J. (1996). Some refinements on the finite-difference
  method for 3-D dc resistivity modeling. *Geophysics*, 61(5), 1301–1307.
- Zhou, B., Dahlin, T. (2003). Properties and effects of measurement errors
  on 2D resistivity imaging surveying. *Near Surface Geophysics*, 1(3),
  105–117.

Standards and policies cited: SPEC 0 (minimum supported dependencies) and
SPEC 7 (seeding pseudo-random number generation) of the Scientific Python
project; PEP 639 and PEP 735; Keep a Changelog 1.1.0; Contributor Covenant
3.0; the JOSS submission requirements and AI usage policy.
