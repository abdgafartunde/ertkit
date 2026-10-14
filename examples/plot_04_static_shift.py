"""
Static shifts between MN segments
=================================

When the potential electrodes are moved apart partway through a Schlumberger
sounding, the curve often shows a step between the two segments, caused by
small near-surface heterogeneities around the electrodes. ertkit can estimate
a multiplicative shift for each segment along with the model.
"""

# %%
# Data with a shifted second segment
# ----------------------------------
# The two segments overlap between AB/2 = 10 m and 14 m, as field crews arrange
# to check the step between segments.

import numpy as np

import ertkit
from ertkit import plot, ves

truth = ves.LayeredModel(resistivity=[200.0, 30.0, 2000.0], thickness=[1.5, 12.0])
short = np.geomspace(2.5, 14.0, 10)
long = np.geomspace(10.0, 300.0, 14)
survey = ves.Survey.schlumberger(
    ab2=np.concatenate([short, long]),
    mn2=np.concatenate([np.full(short.size, 0.5), np.full(long.size, 2.0)]),
)
shift = np.where(survey.mn2 > 1.0, 1.2, 1.0)
clean = ves.forward(truth, survey) * shift
rng = np.random.default_rng(5)
data = ves.Sounding(
    survey=survey,
    rhoa=clean * np.exp(0.03 * rng.standard_normal(survey.n_readings)),
    error=ertkit.ErrorModel(relative=0.03),
)

# %%
# Without and with a static-shift factor
# --------------------------------------
# Ignoring the shift gives a poorer fit; ertkit warns when the goodness-of-fit
# p-value falls below 0.01:

import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    plain = ves.invert_few_layer(data, n_layers=3, rng=0)
print(f"no shift: p = {plain.p_value:.4f}")
for w in caught:
    print("warning:", w.message)

# %%
shifted = ves.invert_few_layer(data, n_layers=3, static_shift=True, rng=0)
print(f"with shift: p = {shifted.p_value:.2f}")
print(
    f"estimated shift of the second segment: {shifted.static_shift[1]:.3f} "
    f"(x/ {shifted.factor[-1]:.3f}); simulated shift {shift.max():.2f}"
)

# %%
plot.sounding(data, plain, shifted, labels=["no shift", "with static shift"])
