"""
Few-layer inversion with uncertainty
====================================

Simulate a Schlumberger sounding over a three-layer earth, invert it for a
three-layer model, and report how well each parameter is determined.
"""

# %%
# A three-layer earth and a sounding
# ----------------------------------
# Resistive topsoil over a conductive weathered layer, on resistive bedrock.
# The sounding uses AB/2 from 2.5 m to 200 m, with MN/2 = 0.5 m for the short
# spacings and 5 m for the long ones, keeping MN at most AB/5 as is usual in
# the field.

import numpy as np

import ertkit
from ertkit import plot, ves

truth = ves.LayeredModel(resistivity=[150.0, 30.0, 800.0], thickness=[2.0, 15.0])
ab2 = np.geomspace(2.5, 200.0, 22)
survey = ves.Survey.schlumberger(ab2=ab2, mn2=np.where(ab2 < 25, 0.5, 5.0))

# Errors of 3% on the transfer resistance, the default, which includes a
# modeling-error floor.
noise = ertkit.ErrorModel(relative=0.03)
data = ves.Sounding.synthetic(truth, survey, noise, rng=5)

# %%
# Inversion
# ---------
# Ten random starting models are tried and the best fit kept.

fit = ves.invert_few_layer(data, n_layers=3, rng=0)
print(
    f"chi2 = {fit.chi2:.1f} for {fit.chi2_target:.0f} degrees of freedom (p = {fit.p_value:.2f})"
)

# %%
# Each parameter with its linearized uncertainty, as a multiplicative factor
# (one standard deviation), next to the true value. "x/ 1.04" means "multiply
# or divide by 1.04".

true_values = np.concatenate([truth.resistivity, truth.thickness])
estimates = np.concatenate([fit.model.resistivity, fit.model.thickness])
for name, est, factor, true in zip(
    fit.parameter_names, estimates, fit.factor, true_values, strict=True
):
    print(f"{name:>6}: {est:8.2f}  (x/ {factor:.3f})   true {true:8.2f}")

# %%
# The depths to the interfaces, which matter most for siting a borehole:

for i, (z, dz) in enumerate(zip(fit.depths, fit.depth_std, strict=True), start=1):
    print(f"bottom of layer {i}: {z:6.2f} m +/- {dz:.2f} m")

# %%
# The data and the fitted curve, then the model with its uncertainty:

plot.sounding(data, fit, labels=["three-layer fit"])

# %%
plot.layered_model(truth, fit, labels=["true model", "three-layer fit"])
