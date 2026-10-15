"""
When the data cannot decide: a suppressed layer
===============================================

A fractured zone between a weathered layer and fresh basement is often the
target of a borehole, yet a sounding may barely see it. This example shows how
the uncertainty reported by ertkit reveals that, instead of hiding it behind a
single best-fitting model.
"""

# %%
# Four layers, one of them suppressed
# -----------------------------------
# The fractured zone (third layer) has a resistivity between those of the
# layers above and below it, so its effect on the sounding curve is small.

import numpy as np

import ertkit
from ertkit import plot, ves

truth = ves.LayeredModel(
    resistivity=[150.0, 40.0, 400.0, 3000.0], thickness=[2.0, 15.0, 25.0]
)
ab2 = np.geomspace(1.0, 500.0, 32)
# Three MN segments, each keeping MN at most AB/5.
mn2 = np.select([ab2 < 10, ab2 < 100], [0.2, 2.0], default=20.0)
survey = ves.Survey.schlumberger(ab2=ab2, mn2=mn2)
data = ves.Sounding.synthetic(truth, survey, ertkit.ErrorModel(relative=0.03), rng=3)

# %%
# Two fits that both match the data
# ---------------------------------
# A four-layer fit, started from the true model so that the search settles on
# the true structure, and a three-layer fit that omits the fractured zone.
# (With random starting models the four-layer search may settle on a
# different structure that fits equally well: another face of the same
# problem.)

four = ves.invert_few_layer(data, n_layers=4, start=truth)
three = ves.invert_few_layer(data, n_layers=3, rng=0)
for name, fit in (("four layers", four), ("three layers", three)):
    print(
        f"{name}: chi2 = {fit.chi2:5.1f} for {fit.chi2_target:.0f} degrees of freedom, p = {fit.p_value:.2f}"
    )

# %%
# Both fits are acceptable. Even knowing the true structure, the four-layer
# fit reports, with a warning, how poorly the data determine the fractured
# zone: large multiplicative uncertainties on its resistivity (``rho_3``) and
# thickness (``h_3``), while the layers above remain well determined.

for name, factor in zip(four.parameter_names, four.factor, strict=True):
    print(f"{name:>6}: known within a factor of {factor:.2f}")

# %%
# The depth to fresh basement, the usual drilling target, is 42 m in the true
# model. The four-layer fit reports that the data barely determine it. The
# three-layer fit gives a precise depth, but to the top of the fractured zone,
# which it merges with the basement: a precise answer to a different question.

for name, fit in (("four layers", four), ("three layers", three)):
    print(f"{name}: basement at {fit.depths[-1]:.1f} +- {fit.depth_std[-1]:.1f} m")

# %%
# The fitted curves are almost indistinguishable:

plot.sounding(data, four, three, labels=["four layers", "three layers"])

# %%
plot.layered_model(
    truth, four, three, labels=["true model", "four layers", "three layers"]
)

# %%
# A linearized uncertainty is only a first warning here: for equivalent
# models it can still understate the range of models that fit. An equivalence
# analysis that samples that range directly is planned for release 0.2.
