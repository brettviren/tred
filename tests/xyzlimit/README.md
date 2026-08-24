# Memory scan for the sliced raster path

Measures what `xyz_limit` buys on a real event, complementing the synthetic
measurement in `tests/regression/qeff_memory_profile.py`. See tred-02x.

## Running

From the **repo root**, with the data files described in `data/README.org`:

    for cfg in tests/xyzlimit/config_unsliced.yaml tests/xyzlimit/config_xlimit*.yaml; do
      tred -c $cfg -l $(basename $cfg .yaml).log fullsim \
        -i data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5 \
        -o $(basename $cfg .yaml).npz
    done

Then `plot_mem_vs_desired.py` and friends parse the logs.

`data/` is gitignored, so the inputs are not in the repo.

## Notes on the port

These came from `tests/playground/zero_out_wf/` on the `mem_xyzlimit` branch
and needed three changes:

- Paths were absolute into the author's home directory
  (`/home/yousen/Public/...`, `/home/yousen/Documents/...`). Now relative to
  the repo root.
- The config key was `mem_xyzlimit`; it is `xyz_limit` here, matching the
  argument name on `Raster` and `compute_qeff`.
- The branch's limits were `[10, 10000, 10000]`, i.e. per-axis with two axes
  effectively disabled. A scalar applies to every axis, and only the largest
  over-limit axis is sliced, so a scalar is the simpler way to express the same
  intent.

## Expected

From the synthetic profile, peak RSS above the torch-import baseline on a long
diffused step (box 66x46x233):

| `xyz_limit` | peak RSS |
| --- | ---: |
| off | 1352 MB |
| 100 | 624 MB |
| 50 | 362 MB |
| 25 | 196 MB |
| 10 | 112 MB |

Wall time is flat across that range. On a GPU the figure that matters is
`torch.cuda.max_memory_allocated()`, not host RSS; re-measure there.
