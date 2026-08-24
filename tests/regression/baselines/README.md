# Regression baselines

Recorded reference outputs for `tests/regression/`.

## `*.pt` and `metadata.json`

Numerical fixtures for the cases in `baseline_cases.py`, asserted by
`test_regression_baseline.py`. Record or re-record with:

    python tests/regression/baseline_cases.py --write [case_prefix]

Re-recording is deliberate. A diff in these files is the reviewable evidence
that behaviour changed; it should accompany the code change that caused it.

## `qeff_memory_profile.json`

The "before" memory profile for the `mem_xyzlimit` port (tred-02x), produced by

    python tests/regression/qeff_memory_profile.py --json

Unlike the `.pt` fixtures this is a **record, not an assertion**. Wall times
and RSS depend on the machine. Regenerate on your own hardware before using it
as a comparison point.

Recorded on: CPU only, torch 2.13.0+cpu, grid spacing [0.4, 0.4, 0.05].
Baseline RSS after importing torch alone: ~244 MB, which the
`rss_over_import_bytes` column subtracts out.

### What it showed

| case | charge box | est peak | measured over import |
| --- | --- | ---: | ---: |
| point_like | 3x3x7 | 0.02 MB | 21.7 MB |
| line | 8x5x17 | 0.18 MB | 22.6 MB |
| wide_diffusion | 9x9x37 | 0.79 MB | 28.0 MB |
| batch_32_steps | 6x5x19 x32 | 4.82 MB | 57.3 MB |
| very_wide_diffusion | 32x31x121 | 31.69 MB | 259.2 MB |
| long_track | 56x36x185 | 98.46 MB | 737.3 MB |
| long_track_diffuse | 66x46x233 | 186.75 MB | 1335.6 MB |

A **single** long, well-diffused step peaks at over 1.3 GB. That was the
problem tred-02x existed to solve.

### After the sliced path (tred-02x.7)

`xyz_limit` bounds peak memory by the slab rather than the whole box. Measured
the same way, peak RSS above the torch-import baseline:

| case | unsliced | `xyz_limit=25` | reduction |
| --- | ---: | ---: | ---: |
| long_track_diffuse | 1352.9 MB | 197.2 MB | 85% |
| long_track | 731.1 MB | 147.5 MB | 80% |
| very_wide_diffusion | 260.0 MB | 89.5 MB | 66% |

Charge totals are identical in every case. Wall time is flat or better --
long_track went from 731 ms to 488 ms, presumably cache locality.

Sweeping the limit on long_track_diffuse:

| `xyz_limit` | peak RSS | time |
| --- | ---: | ---: |
| off | 1352 MB | 880 ms |
| 100 | 624 MB | 824 ms |
| 50 | 362 MB | 870 ms |
| 25 | 196 MB | 916 ms |
| 10 | 112 MB | 823 ms |

Regenerate with `--xyz-limit N`:

    python tests/regression/qeff_memory_profile.py --case long_track_diffuse --xyz-limit 25

Measured peak runs 7.1-8.2x the analytic `field_bytes` across the large cases.
The ratio is stable, so the analytic figure is a usable deterministic proxy
even though it is not the absolute number.

Peak scales as `nbatch * prod(npoints) * prod(box_shape)`, so it grows with
diffusion width (bigger box), step length (bigger box), quadrature order, and
batch size. Note `npoints_4` costs 8x `line` for the same geometry.

### Charge sums here also record tred-xsj

The `charge_sum` column shows the charge-conservation failure: `point_like`
returns 715.83 of 1000 and `anisotropic_z` returns 534.65 of 1000, while
`wide_diffusion` correctly returns 996.35. Any comparison of the ported code
against these numbers is a comparison against an already-wrong reference in
the narrow-sigma regime. That is tred-xsj's problem, not the port's.
