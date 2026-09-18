# Coverage export semantics after R1

The selector, masks, loss, teacher and risk path are unchanged. Raw `core.coverage`
records in GRADIENT_DIAGNOSTICS and PUBLIC_RESULTS remain untouched. In
COVERAGE_AND_RISK.points[].coverage[], reporting adds explicit partition fields.

| Field | Meaning | Denominator |
|---|---|---|
| geometry | Legal geometry pixel count | — |
| fine / coarse | Disjoint selected valid counts | — |
| total_pixels | geometry + legacy ignore | — |
| invalid_geometry_pixels | Original ignore, outside legal geometry | — |
| invalid_geometry_fraction | Invalid geometry fraction | total_pixels |
| unselected_valid_pixels | geometry - fine - coarse | — |
| unselected_valid_fraction | Valid pixels not selected for pseudo supervision | geometry |
| fine_fraction / coarse_fraction | Selected valid fractions | geometry |
| ignore / ignore_fraction | Preserved legacy invalid geometry count/fraction | total_pixels |

When geometry=0 all legal-domain fractions are null, including fine/coarse/unselected.
When total_pixels=0 the normalized invalid geometry fraction is null. Counts must be
nonnegative integers and fine+coarse cannot exceed geometry. Identities are checked:
`fine + coarse + unselected_valid_pixels = geometry` and
`geometry + invalid_geometry_pixels = total_pixels`.
The original record has no independent total-size field: total_pixels is derived from
the raw geometry/ignore partition, not a new measurement of image dimensions.

`legacy_ignore_meaning` labels old fields explicitly. Old ignore is never the valid
rejection rate. All-legal, all-unselected pixels have invalid_geometry_fraction=0,
unselected_valid_fraction=1. All-invalid geometry has invalid_geometry_fraction=1,
unselected_valid_fraction=null. Historical A0 diagnostics remain NOT_MEASURED_HISTORICAL.
No training decision or risk weight uses these export-only transformations.
