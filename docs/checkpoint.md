# Session checkpoint

Written to hand this work over to a GPU machine. Covers what is done, what is
deliberately left, and what specifically needs re-measuring on a GPU.

Companion documents: `docs/branches.md` (what every remote branch contains) and
the beads issue tracker (`bd list`, `bd show <id>`). Every task carries a
closing note with what was found; those notes are the detailed record and this
file only summarises.

## Where things stand

Two merge epics were planned against the original master (`0d439e9`):

| epic | subject | state |
| --- | --- | --- |
| `tred-4pz` | merge `end_to_end` | **closed**, all 17 tasks |
| `tred-02x` | port `mem_xyzlimit` | 12 of 13 tasks closed; one needs a GPU |

Test suite went from **146 passing at `0d439e9` to 319**, with the same 8
pre-existing failures throughout (tracked as `tred-5jz`: three real test bugs,
five missing data files).

Nothing has been force-pushed or rebased. `master-premerge` and `master-origin`
both still point at `0d439e9`; `provisional/merges` holds the eight throwaway
merges from the first session.

## Setting up on the GPU box

The venv here is CPU-only and is **not** what you want there:

```
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install torch            # CUDA build, not +cpu
VIRTUAL_ENV=.venv uv pip install numpy scipy pytest h5py matplotlib pyyaml k3d hist
VIRTUAL_ENV=.venv uv pip install -e . --no-deps
```

`k3d` and `hist` are needed only because `tred.plots.__init__` eagerly imports
every submodule — see `tred-lwd`, still open. `matplotlib` wants a writable
`MPLCONFIGDIR`.

Run the suite with plain `pytest tests/`; `pyproject.toml` now sets
`--import-mode=importlib`, without which collection fails outright on
duplicate test basenames. `-m "not slow"` skips the few multi-second ones.

## What must be re-measured on the GPU

Every number in this work is **host RSS on CPU**. On a GPU the constraint is
device memory, the allocator behaves differently, and the conclusions could
move. Three specific items:

1. **`tred-02x.13`, the only open task.** `@torch.jit.script` on
   `qline_diff3D_script` measured worth nothing on CPU (97.9 ms scripted
   against 98.9 ms plain) and torch warns it is deprecated. TorchScript may
   still matter on GPU. Re-run the benchmark before removing it.

2. **`xyz_limit` defaults.** The slab limit currently defaults to `None`,
   never slicing. The CPU sweep gave 1352/624/362/196/112 MB at
   off/100/50/25/10 with flat wall time, but the right production default
   should be chosen from `torch.cuda.max_memory_allocated()`, not RSS.
   `tests/xyzlimit/` has the configs and scripts; the scan has **not** been
   run on real data.

3. **`TRED_COMPILE_RASTER`.** Opt-in `torch.compile` of the charge kernels
   halved peak-memory scaling and was ~1.7x faster on CPU. Inductor's GPU
   backend is a different code path; measure before considering it as a
   default. Note `dynamic=True` is essential — with `dynamic=False` every new
   charge-box shape triggers a 2.6-4.2 s recompile.

## Running the end-to-end example

`data/` is gitignored, so the response files and the 1.7 GB EDEPSIM input are
not in the repo; `data/README.org` describes them.

```
tred -c tests/fullsim/config_10x10.yaml -l output.log fullsim \
  -i data/MicroProdN1p1_NDLAr_1E18_RHC.convert2h5.nu.0000001.EDEPSIM.hdf5 \
  -o test_output.npz
```

`tests/fullsim/config_10x10.yaml` is the config from `data/README.org` with
paths rewritten relative to the repo root. Run from the repo root.

**This works on CPU now but is slow** — a single event was still working
through the TPCs after twelve minutes, at roughly 190 s per batch. It was left
running rather than completed. On the GPU it should be the natural first
smoke test.

Two things found while getting it to run:

- `graph_effq` hardcoded `runit('cuda')`, so `fullsim` could not run at all
  without a GPU. Now `device: auto` in the config, resolving to CUDA when
  available (`tred-rro`).
- `convo_o_shape` is read from config and never used; `LacedConvo`'s shape is
  hardcoded to `(8, 8, 2048)`, which is *larger* than the `(4, 4, 2048)` the
  documented example asks for. Filed as `tred-n08`, **not fixed**.

## The two branches

### `end_to_end` — absorbed

Every src change is landed or explicitly declined, with a note in the source at
the point of definition so the declines are not re-applied. The full table is
on the `tred-4pz` epic.

The premise was that six global default flips all had to become opt-in. Only
**one** was load-bearing: `to_tensor()` forcing `requires_grad=False`, which
silently detached a grad-requiring leaf and raised on a non-leaf. With that
fixed, a lifetime gradient reaches the loss through the whole `Sim` on
otherwise untouched defaults, and `param()`, `Drifter.lifetime` and the
`absorb()` integer cast all stay as they were.

The pruning in `chunking.accumulate()` was the subtle one, and it needed two
corrections before it was right. It does **not** block gradients. But it is a
data-dependent mask, so moving a fitted parameter changes the surviving block
count (140/142/144 at lifetime 2/10/50) and a loss against a fixed truth fails
on shape. Hence `accumulate(prune=...)`, with `graph_opt` passing
`prune=False` and ordinary runs unaffected.

Not verified: the physics results in `tests/optimizer/README.md`. Those need
HDF5 inputs not in the repo. Acceptance was tested on synthetic data — Adam
converges on lifetime from both 12.4 and 8.7 toward 10.0.

### `mem_xyzlimit` — ported, not merged

The branch's code does not apply: master's `eval_qeff` was rewritten out from
under it (no chunk loop at all, a new point/line model split, different
padding, and an interpolation stencil that widened from 2 to 3, which breaks
the branch's index arithmetic outright).

Re-derived instead. `xyz_limit` evaluates, convolves and places one slab at a
time along the largest over-limit axis, with a halo so the convolution sees its
true neighbours. Exact, not approximate: charge totals are bit-identical and
field values agree to under one ULP. Gradients survive it.

| case | unsliced | `xyz_limit=25` |
| --- | ---: | ---: |
| long_track_diffuse | 1352.9 MB | 197.2 MB |
| long_track | 731.1 MB | 147.5 MB |
| very_wide_diffusion | 260.0 MB | 89.5 MB |

Two branch defects were avoided and are covered by tests: the hardcoded stencil
width, and slicing the first over-limit axis rather than the largest.

## Bugs found and fixed along the way

None of these came from either branch.

- **`tred-xsj`** — `compute_qeff` silently lost charge when sigma was small
  against the grid: 28% at a quarter of a cell, 97.5% at an eighth, exactly
  zero below that, NaN at sigma 0. Fixed with a half-cell floor
  (`min_sigma`). Worth knowing: the error *oscillates* with sub-cell position,
  over- as well as under-estimating, so a single test position hides it.
- **`tred-4pz.6`** — `diffuse()` collapsed an initial sigma to zero for
  negative dt, feeding the above. Now clamps dt. This changed *documented*
  behaviour; the reasoning and BV's confirmation are on the issue, and
  `tred-djb` records what to do if finite-sigma depos are ever backed up.
- **`tred-4pz.2`** — `import tred.bsb` raised `ImportError`;
  `tests/test_bsb.py` had been silently uncollectable.
- **`tred-4pz.3`** — `torch.cuda.reset_peak_memory_stats()` unguarded inside
  the batch loop killed CPU runs on the first batch.

## Open issues

```
tred-02x     epic, pending tred-02x.13
tred-02x.13  remove @torch.jit.script  (needs GPU measurement)
tred-a8k     sequencing decision, keep for the record
tred-n08     convo_o_shape read but never used
tred-lwd     tred.plots eagerly imports everything, so the CLI needs k3d
tred-5jz     3 real test bugs + 5 missing-data failures + duplicate basenames
tred-3ra     pointq's fullsim CLI hijack, for when that branch is merged
tred-fgh     skippad is a dead option aliasing recenter
tred-djb     finite-sigma backup semantics, deferred
```

## Measurement traps worth remembering

Each of these cost a wrong conclusion before being caught:

- `tracemalloc` is **blind to torch CPU tensors** — it reports ~0 for a 400 MB
  tensor. Use RSS.
- `ru_maxrss` in a forked child **inherits the parent's high-water mark**, so a
  parent that has already allocated poisons every later child measurement. The
  profiler keeps all allocation in the children for this reason.
- Absolute RSS at one problem size is dominated by fixed overheads —
  inductor's own footprint made `torch.compile` look *worse*. Measure the
  **slope** against problem size.
- "A gradient reaches the loss" is strictly weaker than "an optimiser can run".
