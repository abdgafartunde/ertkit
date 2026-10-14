# API reference

The public API is what this page documents. Until release 1.0 it may change
between minor releases; the [changelog](changelog.md) records every change.

```{eval-rst}
.. currentmodule:: ertkit

.. autosummary::
   :toctree: generated

   show_versions
   ErrorModel
   ConvergenceWarning
   DataWarning
   results.InversionResult
```

## Vertical electrical sounding

```{eval-rst}
.. currentmodule:: ertkit.ves

.. autosummary::
   :toctree: generated

   LayeredModel
   Survey
   Sounding
   read_csv
   write_csv
   invert_few_layer
   LayerFit
   forward
   potential
   resistivity_transform
   hankel_j0
```

## Electrode arrays

```{eval-rst}
.. currentmodule:: ertkit.arrays

.. autosummary::
   :toctree: generated

   geometric_factor
```
