# Groundwater Detection

Electrical Resistivity Tomography (ERT) inversion for groundwater mapping.

**Status:** Under development ⸺ expected Q2 2027.

## Planned

- 2-D ERT forward modelling with P1 finite elements
- Classical regularised inversion (Tikhonov, TV) for resistivity imaging
- Depth-of-investigation analysis and sensitivity maps
- Synthetic aquifer models with conductive/resistive anomalies

## Motivation

Electrical Resistivity Tomography is a geophysical imaging technique closely
related to EIT, used to map subsurface resistivity distributions for groundwater
exploration, contaminant monitoring, and geotechnical site characterisation.
While `eitkit` targets medical/industrial EIT on the unit disc, this project
extends the same mathematical framework to half-space ERT geometries with
surface electrode arrays and larger spatial scales.

## References

- Loke, M.H. & Barker, R.D. (1996). Rapid least-squares inversion of apparent
  resistivity pseudosections by a quasi-Newton method. *Geophysical
  Prospecting*, 44(1), 131--152.
- Daily, W. et al. (2004). Electrical resistance tomography. *The Leading
  Edge*, 23(5), 438--442.
