# Vertical electrical sounding

This page explains what `ertkit.ves` computes and how to read its results.
The [examples](../auto_examples/index.rst) show each step in code, and the
[verification report](../verification.md) lists the checks behind every
claim made here.

## The layered earth

A vertical electrical sounding is interpreted with a horizontally layered
earth: layer $i$ has resistivity $\rho_i$ and thickness $h_i$, and the bottom
layer is a half-space. In ertkit this is a
{class}`~ertkit.ves.LayeredModel`. Its `depths` are the depths to the bottoms
of the upper layers, and its `conductance` $S_i = h_i/\rho_i$ and
`transverse_resistance` $T_i = h_i \rho_i$ are the Dar Zarrouk parameters
(Maillet 1947), which matter for equivalence (below).

## Forward model

A current $I$ injected at a point on the surface produces the potential

$$
V(r) = \frac{I}{2\pi} \int_0^\infty \mathcal{T}_1(\lambda) \thinspace J_0(\lambda r) \thinspace d\lambda
$$

at distance $r$, where the resistivity transform $\mathcal{T}_1$ follows from
the recurrence

$$
\mathcal{T}_N = \rho_N, \qquad \mathcal{T}_i = \frac{\mathcal{T}_{i+1} + \rho_i \tanh(\lambda h_i)}{1 + \mathcal{T}_{i+1} \tanh(\lambda h_i)/\rho_i}
$$

(Pekeris 1940; Koefoed 1979). The integral is evaluated with the 120-point
digital linear filter (Ghosh 1971) of Guptasarma and Singh (1997), whose
relative error stays below about $10^{-6}$ for resistivity contrasts up to
$10^4$ between layers. The filter values come from
[libdlf](https://github.com/emsig/libdlf) and are licensed CC BY 4.0.

A reading with current electrodes $A$, $B$ and potential electrodes $M$, $N$
gives the apparent resistivity

$$
\rho_a = K \frac{\Delta V}{I}, \qquad K = \frac{2\pi}{1/AM - 1/BM - 1/AN + 1/BN},
$$

where $K$ is the geometric factor ({func}`ertkit.arrays.geometric_factor`).
For a homogeneous earth $\rho_a$ equals its resistivity. A
{class}`~ertkit.ves.Survey` stores $AB/2$ and $MN/2$ for every reading, so
Schlumberger soundings whose $MN$ changes partway, and Wenner soundings, are
modeled exactly, including the finite length of $MN$.

## Errors and misfit

Measurement errors are specified on the transfer resistance $R = \Delta V/I$
by an {class}`~ertkit.ErrorModel`:

$$
s_R = a + b \thinspace |R|,
$$

with an absolute part $a$ in ohms and a relative part $b$ (LaBrecque et al.
1996). The default, 3% relative, is a common choice that includes a floor for
modeling errors: no layered model describes a real site exactly, so field data
should not be fitted more closely than such errors allow. Error models fitted
to reciprocal measurements (Tso et al. 2017) can justify other values.

The inversions work with $d_i = \ln \rho_{a,i}$, whose standard deviations are
$s_i = b + a/|R_i|$, and measure the misfit by

$$
\chi^2 = \sum_{i=1}^{N_d} \left( \frac{d_i - f_i(m)}{s_i} \right)^2.
$$

A {class}`~ertkit.ves.Sounding` removes readings that cannot be inverted in
log space ($R \le 0$, or $|R|$ below three times $a$) and warns about them.

## Few-layer inversion

{func}`~ertkit.ves.invert_few_layer` estimates the logarithms of the layer
resistivities and thicknesses of a model with a chosen number of layers,
starting from several random models. It returns a
{class}`~ertkit.ves.LayerFit` with:

- **Linearized uncertainty.** For each parameter, the standard deviation of its
  logarithm and the corresponding multiplicative factor: a factor of 1.2 means
  the parameter is known within a factor of 1.2 (one standard deviation).
  Depths to the interfaces come with standard deviations in meters.
- **Goodness of fit.** A model that fits the data to their errors has
  $\chi^2 \approx N_d - p$ for $p$ parameters; the `p_value` measures how
  unusual the misfit is. Below 0.01, ertkit warns: either the errors are
  underestimated or the model needs more layers.
- **Unresolved parameters.** A parameter that ends at the edge of the range
  the data could possibly resolve is listed in `at_bound`, with a warning.
- **Static shifts.** With `static_shift=True`, a multiplicative factor for
  each $MN$ segment after the first is estimated with the model.

The linearized uncertainty is honest when the parameters are well resolved:
over many noise realizations, the one-standard-deviation intervals contain the
true values about 68% of the time (see the verification report).

### Equivalence

Some layers cannot be resolved individually. A thin conductive layer is
determined mainly through its conductance $S = h/\rho$, a thin resistive layer
through its transverse resistance $T = h\rho$ (Maillet 1947), and a layer
whose resistivity lies between those of its neighbors can be almost invisible
(suppression). Different models then fit the data equally well. The fit
reveals this through large uncertainty factors, but a linearized uncertainty
can still understate the range of acceptable models. Release 0.2 adds an
equivalence analysis that samples that range directly. The
[equivalence example](../auto_examples/plot_03_equivalence.rst) shows a
fractured zone in crystalline basement that the data barely constrain.

## Smooth inversion and depth of investigation

{func}`~ertkit.ves.invert_smooth` avoids choosing the number of layers. It
uses many thin layers whose thicknesses grow with depth and finds the
smoothest model that fits the data to their errors (Occam's inversion;
Constable et al. 1987): the regularization weight $\mu$ is the largest with
$\chi^2 = N_d$. If even the smoothest model fits better than that, ertkit
reports it and warns that the errors may be overestimated.

The result, a {class}`~ertkit.ves.SmoothFit`, includes:

- **Sensitivity.** For each layer, how strongly the data respond to its
  resistivity, weighted by the data errors.
- **Depth of investigation.** Two further inversions pull the model toward
  very different reference resistivities, 0.1 and 10 times the geometric mean
  of the apparent resistivities. Where the data control the model the two
  agree; where they do not, each follows its reference. The index
  $R^{\mathrm{DOI}} = (m^{(1)} - m^{(2)})/(m_{\mathrm{ref}}^{(1)} - m_{\mathrm{ref}}^{(2)})$
  (Oldenburg and Li 1999; Marescot et al. 2003) is rescaled to 1 at the
  bottom. The depth of investigation is the shallowest depth from which the
  magnitude of the index stays at or above 0.1 (`ertkit.ves.DOI_THRESHOLD`)
  over at least a factor of 1.5 in depth; below it, the model reflects the
  regularization, not the data. Plots shade that region, and also any single
  shallower layer whose index reaches 0.1. If the two reference
  inversions still agree at the bottom of the model, ertkit warns and reports
  no depth of investigation; a larger `depth_range` then helps.

## Limitations

- The earth is assumed flat and horizontally layered. Lateral changes,
  topography and a dipping contact all distort a sounding.
- Electrical anisotropy cannot be separated from layer thickness in VES: only
  the geometric mean of the horizontal and vertical resistivities, and a
  thickness scaled by the coefficient of anisotropy, are resolved, so depths
  can come out too large.
- Errors in electrode positions bias apparent resistivities (Zhou and Dahlin
  2003).
- A low resistivity can mean water-bearing material, clay, or both. The
  inversion cannot tell these apart; that takes geological knowledge.

## References

- Binley, A., Slater, L. (2020). *Resistivity and Induced Polarization:
  Theory and Applications to the Near-Surface Earth*. Cambridge University
  Press.
- Constable, S. C., Parker, R. L., Constable, C. G. (1987). Occam's inversion:
  a practical algorithm for generating smooth models from electromagnetic
  sounding data. *Geophysics*, 52(3), 289–300.
- Ghosh, D. P. (1971). The application of linear filter theory to the direct
  interpretation of geoelectrical resistivity sounding measurements.
  *Geophysical Prospecting*, 19(2), 192–217.
- Guptasarma, D., Singh, B. (1997). New digital linear filters for Hankel J0
  and J1 transforms. *Geophysical Prospecting*, 45(5), 745–762.
- Koefoed, O. (1979). *Geosounding Principles 1: Resistivity Sounding
  Measurements*. Elsevier.
- LaBrecque, D. J., Miletto, M., Daily, W., Ramirez, A., Owen, E. (1996). The
  effects of noise on Occam's inversion of resistivity tomography data.
  *Geophysics*, 61(2), 538–548.
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
- Tso, C.-H. M., Kuras, O., Wilkinson, P. B., Uhlemann, S., Chambers, J. E.,
  Meldrum, P. I., Graham, J., Sherlock, E. F., Binley, A. (2017). Improved
  characterisation and modelling of measurement errors in electrical
  resistivity tomography (ERT) surveys. *Journal of Applied Geophysics*, 146,
  103–119.
- Zhou, B., Dahlin, T. (2003). Properties and effects of measurement errors
  on 2D resistivity imaging surveying. *Near Surface Geophysics*, 1(3),
  105–117.
