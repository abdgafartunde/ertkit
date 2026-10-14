"""
Smooth inversion and depth of investigation
===========================================

Invert a sounding without choosing the number of layers, and find the depth
below which the model no longer reflects the data.
"""

# %%
# Data over a weathered crystalline basement
# ------------------------------------------
# Topsoil, a clay-rich weathered layer and fresh basement: a common setting
# for groundwater in Nigeria's basement complex.

import numpy as np

import ertkit
from ertkit import plot, ves

truth = ves.LayeredModel(resistivity=[200.0, 30.0, 2000.0], thickness=[1.5, 12.0])
survey = ves.Survey.schlumberger(ab2=np.geomspace(2.5, 150.0, 24), mn2=0.5)
data = ves.Sounding.synthetic(truth, survey, ertkit.ErrorModel(relative=0.03), rng=4)

# %%
# Occam inversion
# ---------------
# Thirty layers whose thicknesses grow with depth. The regularization weight
# is the largest that still fits the data to their errors.

smooth = ves.invert_smooth(data)
print(f"chi2 = {smooth.chi2:.1f} for {smooth.n_data} data ({smooth.stop_reason})")
print(f"depth of investigation: {smooth.doi:.1f} m")

# %%
# The shaded region lies below the depth of investigation: two further
# inversions, pulled towards very different reference models, disagree there,
# so the data do not determine the resistivity at those depths.

ax = plot.layered_model(
    truth, smooth, labels=["true model", "smooth model"], log_depth=True
)

# %%
# The depth of investigation index itself, layer by layer. It is near 0 where
# the data control the model and near 1 where the reference model does.

import matplotlib.pyplot as plt

tops = np.concatenate([[0.0], smooth.model.depths])
fig, ax = plt.subplots(figsize=(4.5, 4.5), layout="constrained")
ax.step(smooth.doi_index, tops, where="post")
ax.axvline(ves.smooth.DOI_THRESHOLD, color="gray", linestyle="--", label="threshold")
ax.set_yscale("log")
ax.invert_yaxis()
ax.set_xlabel("depth-of-investigation index")
ax.set_ylabel("depth (m)")
ax.legend()

# %%
# A longer spread looks deeper:

for ab2_max in (50.0, 150.0, 500.0):
    longer = ves.Survey.schlumberger(ab2=np.geomspace(2.5, ab2_max, 24), mn2=0.5)
    sounding = ves.Sounding.synthetic(
        truth, longer, ertkit.ErrorModel(relative=0.03), rng=4
    )
    print(
        f"AB/2 up to {ab2_max:5.0f} m: depth of investigation {ves.invert_smooth(sounding).doi:6.1f} m"
    )
