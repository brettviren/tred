# Branch survey

A description of every remote branch in this repository as it stood **before any
merging into `master`**, what each one contributes, and what merging it implies.

All comparisons are against `master` at `0d439e9` ("Merge depos regression docs
and test", 2026-04-24), referred to below as *pre-merge master*.  In this working
copy that commit is also available as the branch `master-premerge`.

The branch author's own assessment is that **`end_to_end`** and **`mem_xyzlimit`**
carry the most important changes, that the remaining branches are prototyping
work, and that these two are not compatible with `master` as they stand and
should each be merged separately with care.  This document is organised around
that assessment: those two branches get a detailed treatment first, and the rest
are summarised afterwards.

---

## Summary table

"Ahead" counts non-merge commits unique to the branch relative to its merge-base
with pre-merge master.

| Branch | Base date | Ahead | Touches core `src/tred` | Character |
| --- | --- | ---: | --- | --- |
| **`end_to_end`** | 2025-09-02 | 11 | **yes, deeply** | **Differentiable simulation + lifetime fitting** |
| **`mem_xyzlimit`** | 2025-09-02 | 3 | **yes, deeply** | **Raster memory control by axis chunking** |
| `tred_2x2_analysis` | 2025-07-09 | 28 | yes, additively | LArPix readout models, threshold studies |
| `pgun_signal_processing` | 2025-12-19 | 9 | yes, breaking | Burst readout, noise-free "true" hits |
| `benchmark_accuracy` | 2025-09-02 | 16 | yes | Charge-conservation and quadrature studies |
| `sp` | 2025-07-09 | 18 | yes | 3D deconvolution fitting |
| `pointq` | 2026-04-24 | 5 | new module | Point-charge study driver |
| `fdcomp` | 2025-09-02 | 3 | new module | Far-detector comparison driver |
| `fr_interp` | 2026-04-24 | 2 | yes, small | Field-response interpolation study |
| `test_nd_response` | 2025-12-19 | 4 | yes, small | Comparison against official ND |
| `single_point_charge` | 2025-12-19 | 5 | yes | Single-point-charge hack |
| `benchmark_output_size` | 2025-09-02 | 4 | yes | Output-size benchmark scaffolding |
| `2x2_scaleq_playground` | 2025-07-09 | 12 | yes | Charge scaling for 2x2 |
| `xin_test` | 2025-04-17 | 4 | yes | Abandoned optimisation attempt |
| `benchmark_currents_accuracy` | 2025-12-19 | 1 | yes, small | Waveform comparison |
| `differential` | 2025-09-02 | 1 | yes | Gradient support (subset of `end_to_end`) |
| `sparse_bad_archived_20250323` | 2025-03-21 | 1 | yes | Self-labelled archived |
| `archived20250228` | 2025-01-22 | 3 | additive | Self-labelled archived |
| `2x2_playground` | 2025-07-09 | 0 | — | Already in master |
| `benchmark_playground` | 2025-12-19 | 0 | — | Already in master |
| `playground` | 2025-05-06 | 0 | — | Already in master |
| `refactor` | — | 0 | — | Identical to master |
| `unit_tests` | — | 0 | — | Identical to master |

---

# Part 1 — The two priority branches

## `end_to_end`

*11 commits, forked 2025-09-02.  Includes all of `differential`.*

### Overview

This branch makes the tred simulation chain **differentiable end to end**, so
that a physics parameter can be fitted by gradient descent through the whole
detector simulation, and then demonstrates it by fitting the electron lifetime
against a simulated "truth" waveform.

This is the most conceptually significant work in the repository.  It is not a
study configuration — it changes what the simulation *is capable of*.  The
demonstration (`tests/optimizer/`) fits lifetime on long muon tracks, including
stress configurations and a written-up set of results.

The work divides cleanly into two very different halves, and the distinction
matters a great deal for merging:

- **Additive, opt-in machinery** — new functions and a new `ChunkSum` method
  that exist alongside the old ones and change nothing unless selected.
- **Global default flips** — small edits that change the behaviour of the
  *existing* code paths for every caller, whether or not they want gradients.

The first half is safe.  The second half is what makes the branch incompatible
with `master`.

### What it adds (the safe half)

**`src/tred/sparse.py`** gains two functions, both purely additive:

- `fill_envelope_index_put()` — an autograd-friendly twin of `fill_envelope()`.
  The original fills an envelope by in-place assignment, which breaks the
  autograd graph; this one uses `Tensor.index_put()` and rebinds via
  `set_data()`, so gradients flow back to the source block.
- `chunkify_bins()` — the same as `chunkify()` but routed through
  `fill_envelope_index_put()`.

**`src/tred/graph.py`** gains:

- `to_param()` — wraps a value as a leaf `nn.Parameter`, detaching first so it
  is a proper autograd leaf.
- A new `ChunkSum` method `'chunksum_differential'`, selected by name like the
  existing `chunksum_inplace_v1`/`v2`, implemented as
  `accumulate(chunkify_bins(block, shape))`.

**`tests/sparse/test_gradient.py`** — two tests: one checking `chunkify_bins()`
agrees numerically with `chunkify()`, one checking that gradients propagate
correctly back through `accumulate(chunkify_bins(...))` by comparing against a
hand-built dense reference.

**`src/tred/plots/graph_opt.py`** (462 lines, new) — the `train` command.  Builds
the normal `Sim` graph, replaces `drifter.lifetime` with a trainable parameter,
and runs Adam against an MSE loss on the current waveforms.  It deliberately
starts from a perturbed lifetime (`lifetime_inc = 1.2395…`, `lifetime_dec =
0.8711…`) and fits from both sides to show convergence.

**`tests/optimizer/`** — configs, run scripts, a muon-track generator, plotting
scripts, and two READMEs documenting inputs, outputs and results.

### What it changes globally (the problematic half)

These are the compatibility issues.  Each is a one- or two-line edit whose blast
radius is the entire codebase.

**1. `graph.param()` now creates gradient-tracking parameters by default.**

```python
# before
return nn.Parameter(torch.tensor(thing, dtype=dtype), requires_grad=False)
# after
return nn.Parameter(torch.tensor(thing, dtype=dtype))   # requires_grad=True
```

`nn.Parameter` defaults to `requires_grad=True`, so this flips every value built
through `param()` from a constant into a trainable parameter.  Ordinary forward
simulation then builds and retains an autograd graph it never uses, costing
memory and time.

**2. `Drifter.lifetime` is unconditionally a trainable parameter.**

```python
# constant(self, 'lifetime', lifetime)
self.lifetime = to_param(lifetime, dtype=torch.float32, requires_grad=True)
```

Every `Drifter`, in every run, now carries a trainable lifetime.

Worth noting: **this change appears to be unnecessary even for the branch's own
purpose.**  `graph_opt.train()` explicitly does

```python
sim.charge.drifter.lifetime = nn.Parameter(
    torch.tensor(lifetime*lifetime_inc, device=device), requires_grad=True)
optimizer = optim.Adam([sim.charge.drifter.lifetime], lr=lr)
```

— it installs its own trainable parameter and optimises only that one tensor.
The same reasoning applies to the `param()` flip above.  Both global defaults
could very likely be reverted to `requires_grad=False` with no loss of
functionality, which would remove the bulk of this branch's incompatibility.
**This is the single highest-value thing to check when merging.**

**3. `chunking.accumulate()` no longer drops zero content.**

```python
# indices = torch.where((torch.abs(summed) > 1e-3).any(dim=[...]))
# return Block(location=unique_locs[indices], data=summed[indices])
return Block(location=unique_locs, data=summed)
```

The pruning step is commented out rather than made conditional.  Boolean-mask
indexing is awkward under autograd and produces data-dependent shapes, so
removing it is understandable for the differentiable path — but it also removes
a sparsification step that the ordinary path relies on to keep block counts
down.  **This is a potential memory and throughput regression for every normal
simulation run**, and its size depends on how much all-zero content the chunking
actually produces, which has not been measured here.

Note that `pgun_signal_processing` makes the *identical* edit independently,
which suggests it is a real obstacle and not a one-off hack.

**4. `drift.absorb()` no longer casts charge to integer.**

```python
# charge = charge.to(dtype=torch.int32)
```

Required, since an integer cast has no useful gradient.  But `absorb()` then
does, when `fluctuate=True`:

```python
b = Binomial(charge, loss)
```

with a now-floating-point `total_count`.  The branch adds a `# FIXME: Not
differentiable` comment on the fluctuate path but does not resolve it.  **The
interaction of non-integer charge with `Binomial` should be checked explicitly**
— this affects the physics of the fluctuating path, not just gradients.

**5. `drift.diffuse()` replaces NaN masking with negative-`dt` clamping.**

```python
dt = torch.where(dt<0, torch.zeros_like(dt), dt)
...
# sigma[torch.isnan(sigma)] = 0
```

The in-place NaN assignment breaks autograd, so it is replaced by clamping the
cause (negative `dt` producing `sqrt` of a negative) rather than patching the
symptom.  This is arguably a genuine improvement, but it is a behaviour change:
NaNs arising from any *other* source are no longer silently zeroed and will now
propagate.

**6. `raster/steps.py: to_tensor()` no longer forces `requires_grad = False`.**

Setting `requires_grad = False` on a non-leaf tensor raises under autograd, so
the lines are dropped.  The consequence is that tensors passed into the raster
now keep whatever gradient status they arrived with, rather than being
defensively detached.

### Merge implications

- The additive half (`sparse.py`, `to_param`, `chunksum_differential`,
  `graph_opt.py`, the tests and studies) can be merged with essentially no risk.
  It conflicts with nothing and is inert unless selected.
- The global half needs to become opt-in before it is safe.  Concretely:
  restore `requires_grad=False` as the default in `param()`; restore
  `constant(self, 'lifetime', …)` in `Drifter` and let `graph_opt` install its
  own parameter as it already does; make the `accumulate()` pruning conditional
  rather than deleted; and decide deliberately about the `absorb()` integer cast.
- Textual conflicts are minor.  Merging this branch into pre-merge master
  produced exactly one conflict, in `pyproject.toml`, where the branch promotes
  `h5py`, `hist`, `pyyaml` and `scikit-learn` from optional extras to hard
  dependencies.  That contradicts master's deliberate optional-extras layout and
  should be declined.
- **The low textual conflict count is misleading and is the main trap here.**
  This branch merges almost cleanly while changing global defaults across the
  whole simulation.  It needs review on semantics, not on conflict markers.

> **Status in this working copy:** `end_to_end` has been merged into local
> `master` (commit `ff7f4d2`).  That merge resolved the `pyproject.toml`
> conflict in master's favour, but it **did take all six global changes listed
> above**.  Items 1–4 in particular should be revisited before this is pushed.

---

## `mem_xyzlimit`

*3 commits, forked 2025-09-02: "adapat logic of dynamic chunk", "control mem by
xyz_limit", "benchmark xlimit".*

### Overview

This branch attacks the peak-memory bottleneck in charge rasterisation.

In `raster/steps.py`, `eval_qeff()` evaluates the charge model over the entire
bounding box of a step, reshapes it to `(batch, lmn_prod, D1, D2, D3)`, pads it,
and only then convolves with the quadrature kernel.  Peak memory is therefore set
by the *full* box times the number of quadrature channels, even though the
convolution output is comparatively small.  For long or highly diffused steps
this is the dominant allocation in the whole simulation.

The branch restructures this to **evaluate, convolve and accumulate one slice at
a time along a chosen axis**, so peak memory is bounded by the slice size rather
than the whole box.  A new `xyz_limit` parameter sets the maximum extent per axis
before slicing kicks in, and it is plumbed from `Raster` down through
`raster_steps()` into `eval_qeff()`.

**The core idea is sound and valuable, and the accumulation scheme is correct.**
The problems are all in the surrounding implementation, which is benchmark
scaffolding rather than finished code.

### How the new scheme works

A new helper does the per-slice work:

```python
def eval_qeff_singlerun(Q, X0, X1, Sigma, x, y, z, kernel, lmn, lmn_prod, rst):
    charge = eval_qmodel(Q, X0, X1, Sigma, x, y, z)
    charge = charge.view(Q.size(0), lmn_prod, x.shape[-1], y.shape[-1], z.shape[-1])
    charge = torch.nn.functional.pad(charge, pad=(rst[2]-1, rst[2]-1, ...))
    torch.cuda.empty_cache()
    charge = torch.nn.functional.conv3d(charge, kernel.to(charge.dtype),
                                        padding='valid', groups=lmn_prod)
    return torch.sum(charge, dim=[1,])
```

and `eval_qeff()` loops over slices, scattering each result into a
pre-allocated output:

```python
for j in range(0, xi.size(-1), xchunk):
    jend = min(j + xchunk, xi.size(-1))
    qj = eval_qeff_singlerun(..., xi[..., j:jend], yi, zi, ...)
    indices = torch.arange(j, jend+1, device=device).view(1, -1, 1, 1)
    charge = charge.scatter_add(dim=-3, index=indices.expand_as(qj), src=qj)
```

The overlap arithmetic is right.  Padding by `rst-1` on both sides and
convolving `valid` with a kernel of size `rst` gives an output of
`slice + rst - 1`, so consecutive slices overlap by `rst - 1` elements, and
`scatter_add` correctly sums the overlap.  The output buffer is allocated as
`shape + 1` per axis, matching `shape + rst - 1` for `rst = 2`.

### Problems

**1. It hardcodes an interpolation stencil width of 2, which master no longer uses.**

The index range `torch.arange(j, jend+1)` has length `slice + 1`, while `qj` has
extent `slice + rst[0] - 1`.  These agree only when `rst == 2`.

`rst` is the interpolation stencil width, `kernel.size()[3:]`, and it is *not*
`npoints`.  At the branch's base, `_create_u1d_GL()` returned a `[npt, 2]` tensor
— linear interpolation, stencil width 2 — so the arithmetic was correct there.
**On master the stencil is 3 wide**: `_create_u1d_GL()` now returns `[npt, 3]`
with coefficients `[-r/2, 1+r/2, 0]` / `[0, 1-r/2, r/2]`, from the "point/line
conversion; 3-pt interpolations" work.  So `qj` is `slice + 2` while the index
range is `slice + 1`, and the code raises a shape error against master.

The remedy is the same either way: derive the extent from `qj.shape` rather than
assuming any stencil width.

**2. The `skippad` code path is deleted.**

Pre-merge master has:

```python
if skippad:
    charge = torch.nn.functional.conv3d(charge, kernel, padding='valid', groups=lmn_prod)
    offset = offset + 1
else:
    charge = torch.nn.functional.pad(charge, ...)
    charge = torch.nn.functional.conv3d(charge, kernel, padding='valid', groups=lmn_prod)
```

`eval_qeff_singlerun()` implements only the `else` branch and never applies
`offset = offset + 1`.  But `compute_qeff()` still accepts and normalises the
`recenter` / `skippad` kwargs (setting one implies the other, with a logged
warning).  **Calling with `recenter=True` or `skippad=True` will now silently
produce a wrong offset** — charge displaced by one grid cell, with no error.
This is the most dangerous defect on the branch.

**3. Batch chunking is disabled, not integrated.**

```python
nchunk = int(nbtensor // mem_limit) + 1          # still computed
# chunks = [v.chunk(nchunk, dim=0) for v in [Q, X0, X1, Sigma, x, y, z]]
chunks = [[Q, ], [X0, ], [X1, ], [Sigma, ], [x, ], [y,], [z,]]
```

The pre-existing batch-dimension chunking is commented out, leaving `nchunk`
computed and unused.  So one memory-control mechanism is *removed* while another
is added.  These are complementary — batch chunking bounds the number of steps in
flight, axis chunking bounds the box per step — and both should be available.

**4. Debug scaffolding left in the hot path.**

```python
print(usex, usey, usez, len(chunks[0]), torch.cuda.max_memory_allocated()/1024**2)
```

This sits in the `else` branch, which is the *common, unchunked* path.  It prints
on every call, and `torch.cuda.max_memory_allocated()` is not safe on a CPU-only
build.  Separately, `torch.cuda.empty_cache()` is called inside
`eval_qeff_singlerun()`, i.e. once per slice; it releases the caching allocator's
reserved blocks and will noticeably slow the very loop it sits in.

**5. The chunking trigger changed meaning.**

```python
# xyzchunk = (xyz_limit < shape) & (torch.prod(shape) > shape_limit)
xyzchunk = xyz_limit < shape
# usex, usey, usez = xyzchunk & (torch.arange(3, device=device) == xyzchunkidx)
usex, usey, usez = xyzchunk
```

Two guards are dropped.  The total-size guard (`shape_limit`, still read from
kwargs but now unused) meant small boxes were never sliced regardless of aspect
ratio; without it, slicing triggers on any axis over the limit even when the
total is tiny.  And restricting slicing to the *largest* axis (`xyzchunkidx`) is
gone, so with an `if/elif/elif` cascade the branch now always slices along **x**
if x is over the limit, even when y or z is far larger — which is the opposite of
what you want for memory.

**6. Quadrature-model kwargs are dropped.**

Master's unchunked path calls `eval_qmodel(Qi, X0i, X1i, Sigmai, xi, yi, zi, **kwargs)`.
`eval_qeff_singlerun()` calls it without `**kwargs`, so the `qmodel=` selector
and any model options are silently ignored.

**7. The default `xyz_limit` changes 100 → 1000**, in both `steps.py` and the new
`Raster` default, without discussion.  Combined with the removed `shape_limit`
guard this materially changes when slicing happens.

**8. `Raster` takes the new option through `**kw`.**

```python
def __init__(self, velocity, grid_spacing, pdims=(1,2), tdim=-1, nsigma=3.0, **kw):
    xyz_limit = kw.get("xyz_limit", [1000, 1000, 1000])
```

An untyped catch-all means a misspelled option is silently ignored rather than
raising.  Master's `Raster` now takes explicit `npoints` and `dtype` arguments;
`xyz_limit` should follow that pattern.

### Merge implications

- **The plumbing must be rewritten, not merged.**  The branch changes
  `raster_steps()` to forward `**kwds` wholesale into `compute_qeff()`, while
  master now reads specific keys out of `kwds` (`npoints`, `dtype`) and passes
  them explicitly.  These two designs collide directly: forwarding `**kwds` on
  top of master's explicit `npoints=` would pass the keyword twice and raise
  `TypeError`.  The `xyz_limit` path needs re-plumbing against master's current
  signature.
- **The code this branch modifies no longer exists on master.**  Pre-merge
  master's `eval_qeff()` has no `xyz_limit`, no `nchunk`, no `shape_limit` and
  no chunk loop of any kind — the entire block the branch edits was removed and
  the function rewritten.  It now selects per-step between a point model and a
  line model with `torch.where(too_short_mask, eval_qmodel(qpoint), eval_qmodel(qline))`
  and convolves with `padding='same'` instead of pad-plus-`valid`.  There is
  effectively nothing here to merge; the branch is a source of ideas, not
  patches.
- **A large memory win is available without any chunking.**  That `torch.where`
  evaluates *both* charge models over the *entire* grid and then discards one
  per step, so it costs roughly twice the peak memory and compute of evaluating
  only the needed model.  Fixing that is independent of the slicing work, is
  numerically checkable, and may deliver much of the benefit this branch was
  chasing.  There is also a wasted allocation immediately above it: `charge` is
  filled with `torch.zeros((Q.size(0), *lmn, *shape))` and then overwritten
  wholesale by the `torch.where` result.
- **`raster/steps.py` has moved substantially under this branch.**  Master has
  seven commits on that file since the fork, including "stabilize raster
  charge-box indexing at dtype boundaries", "dtypes in raster functions", and a
  redefined interpolation with new unit tests.  A textual merge produces
  conflicts in `graph.py`, `steps.py` and `graph_effq.py`, and taking the
  branch's side would revert those fixes.
- **Recommended approach: port, don't merge.**  Re-apply the slice-and-accumulate
  idea onto master's current `eval_qeff()` as a new code path, keeping master's
  `skippad` branch and batch chunking intact, deriving slice extents from
  `qj.shape` instead of assuming `rst == 2`, and gating the whole thing behind an
  explicit `xyz_limit` argument that defaults to off.  Cherry-picking
  `eval_qeff_singlerun()` as a starting point is reasonable; merging the branch
  wholesale is not.
- The benchmark material (`tests/playground/zero_out_wf/plot_mem_vs_desired.py`,
  `plot_nseg.py`, `plot_nseg_hist.py`, and the `config_effqmem_xlimit*.yaml`
  configs) is additive and can come across unchanged.  It is the evidence for
  whether the optimisation actually pays off, so it is worth keeping.

> **Status in this working copy:** not merged.

---

# Part 2 — Prototyping branches

These hold study configurations, run scripts and analysis notebooks.  Most also
poke at `src/tred/plots/graph_effq.py`; see the note on that file at the end.

### `tred_2x2_analysis` — LArPix readout models

*28 commits, forked 2025-07-09.  Contains all of `2x2_scaleq_playground`.*

The largest and most recent study branch, and the best-engineered of them.  It
adds a family of LArPix readout models to `src/tred/readout.py` — periodic reset
(`nd_readout_prc`, with an optional synchronised rolling schedule), correlated
OU/AR(1) output noise (`nd_readout_ou`), a reset model (`nd_readout_rst`,
`nd_readout_full`), signal-dependent noise amplitude (`nd_readout_qdep`,
`nd_readout_qdep_corr`), and a separated trigger-side/record-side noise path
(`nd_readout_sep`) — plus helpers `sync_prc_phase`, `apply_periodic_reset` and
`make_ou_noise`.

Crucially, **these are added as siblings of `nd_readout`, which keeps its
existing two-tuple contract unchanged.**  Existing callers are unaffected.  This
is the pattern the other core-touching branches should have followed.

In `graph_effq.py` it adds config-gated `const_recomb`, `scale_q`/`event_dqdx`
charge scaling, a two-step charge-deposition fluctuation (`charge_fluct`, issue
#27), dispatch over the new readout models, and convolution shapes derived from
the response footprint instead of hardcoded.  It also brings the 2026-06/07
threshold-study configs and run scripts.

*Merging:* conflicts in `graph_effq.py` are real semantic ones — master gained
per-stage benchmarking instrumentation that this branch predates and drops — so
they must be combined rather than chosen between.  `readout.py` merges without
conflict.

### `pgun_signal_processing` — burst readout and "true" hits

*9 commits, forked 2025-12-19.*

Adds burst readout (reading `nburst` samples after each trigger), a parallel
noise-free "true" accumulator so hit charge can be compared against truth, a
configurable readout model, and `fixed_interval_readout()` for interval-averaged
readout.  Also a Bee event-display dumper.

*Merging — the main obstacle:* it **changes `nd_readout`'s return from a
two-tuple to a three-tuple**, and changes `hits` from 1-D to 2-D with
`nburst=4` by default.  Only `graph_effq.py` was updated to match.  Master now
also has `graph_pointq.py`, `graph_fd.py` and `graph_opt.py` calling
`nd_readout`, all of which would break at runtime, and the burst dimension
changes hit shape for every caller.  The fix is to follow the
`tred_2x2_analysis` pattern: keep the two-tuple default and make burst and
true-hit output opt-in.  It also makes the same `chunking.accumulate()`
zero-drop removal as `end_to_end`.

### `benchmark_accuracy` — charge conservation and quadrature

*16 commits, forked 2025-09-02.*

Studies of charge conservation and quadrature accuracy: `tests/effq/studies.org`
and `three_pts_interpolation.org` (~1400 lines of written-up results),
`src/tred/raster/steps_mod.py` (a standalone extension module offering
alternative Gauss-Legendre rules and charge-per-quadrant handling, importing from
`steps.py` without modifying it), and accuracy configs and notebooks.

*Merging:* the documentation and `steps_mod.py` are additive and safe.  The core
changes are not: they include a blanket float32 → float64 promotion throughout
`steps.py` and a `Raster` without master's `dtype` plumbing.  Master has since
absorbed both improvements in better form — including this branch's genuine
`assert torch.any(...)` → `torch.all(...)` fix, where the `any()` form made the
bounds check a no-op.

### `sp` — 3D deconvolution fitting

*18 commits, forked 2025-07-09.  Shares ancestry with `tred_2x2_analysis`.*

Signal-processing studies; the distinctive content is `fit_deconv3d.py` and its
notebook — a 3D deconvolution fit with a configurable kernel, L2 regularisation
and a command-line interface.  Also adds an explicit setuptools src-layout block
to `pyproject.toml`, which is a genuine build improvement.

*Merging:* its `graph_effq.py` change comments out the effq drift-time offset
`effq_blocks_l[-1][:,-1] += int(abs(drtoa/velocity)//tspace)`.  That is a
study-local choice about where effq is evaluated, and taking it would silently
shift the time coordinate for every effq consumer.

### `pointq` — point-charge study

*5 commits, forked 2026-04-24 (from master's tip).*

A new 786-line study driver `src/tred/plots/graph_pointq.py`, plus configs and an
analysis script.

*Merging:* mergeable without textual conflict, but it contains a regression —
it **repoints the existing `fullsim` CLI command at `graph_pointq`**, silently
hijacking the main entry point away from `graph_effq`.  It needs its own command
name instead.

### `fdcomp` — far-detector comparison

*3 commits, forked 2025-09-02.*

A new 389-line driver `src/tred/plots/graph_fd.py` and an `fdsim` CLI command,
for running tred against FD depos with multiple wire planes, time filtering and
waveform concatenation.  Additive and clean.

*Note:* the new command's Python function is named `fullsim`, shadowing the
existing one.  Click registers by the string name so it works, but three
commands in `cli.py` end up sharing one function name.

### `fr_interp` — field-response interpolation

*2 commits, forked 2026-04-24 (from master's tip).*

Configs comparing 0.05 µs and 0.1 µs response sampling, plus distribution-overlay
and charge-difference analysis scripts.  Its `graph_effq.py` change **enables the
hit export and the `write_npz()` call, both of which are commented out on
master** — meaning `fullsim` on master currently produces no output file at all.
Small and worth having for that reason alone.

### `test_nd_response` — comparison with official ND

*4 commits, forked 2025-12-19.  Contains all of `benchmark_currents_accuracy`.*

Accuracy configs and two notebooks comparing tred's field response and waveforms
against the official ND simulation.  Its `graph_effq.py` change enables the
`save_waveform`-gated current export and writes the run's configuration
(yamls, lifetime, thresholds, noise settings, timing) into the output npz as
provenance.  Both are useful beyond this study.

*Merging:* it also moves `write_npz()` to the very end of `runit()`, after an
unguarded `torch.cuda.max_memory_allocated()` call — which would lose all output
on a CPU run.  Keep the earlier placement.

### `single_point_charge` — single-point-charge hack

*5 commits, forked 2025-12-19.  Shares the "burst readout" commit with
`pgun_signal_processing`.*

Configs and plotting for a single point charge, plus a memory-usage plotter.
Largely superseded by `pgun_signal_processing`; the readout and `graph_effq`
changes are the same lineage, and only the study directory is unique.

### `benchmark_output_size` — output-size benchmark

*4 commits, forked 2025-09-02.*

Configs for two time-window lengths and a `convo_o_shape` config knob.

*Do not merge as-is.*  It contains

```python
# temp benchmark
if True:
    info(...)
    continue
```

in `runit()`, which unconditionally skips the entire readout stage.  The
`convo_o_shape` knob is in any case superseded by `tred_2x2_analysis`'s
response-footprint-derived shapes.

### `2x2_scaleq_playground` — charge scaling for 2x2

*12 commits, forked 2025-07-09.*

Batch particle-gun run scripts and shield/signal-processing configs, plus charge
scaling in the pgun path and a fix to pixel-index ranges.  Entirely contained in
both `tred_2x2_analysis` and `sp`; no independent merge needed.

### `xin_test` — abandoned optimisation

*4 commits, forked 2025-04-17.*

Optimisation attempts in `convo.py` and `partitioning.py`.  The final commit
message is "add some optimized code, which does not lead to improvement".  Taken
at the author's word, this is a dead end; keep as a record of what was tried.

### `benchmark_currents_accuracy` / `differential`

Single-commit branches, both strict subsets of larger ones —
`benchmark_currents_accuracy` of `test_nd_response`, `differential` of
`end_to_end`.  No independent action needed.

### `sparse_bad_archived_20250323`, `archived20250228`

Self-labelled archived.  The first tightens `Block.shape` to always be a
`torch.Size` and adds blocking unit tests, from a base 138 commits behind.  The
second adds recombination functions, a plotting module, an effective-charge
profiler and a 1037-line `toy_gpu3.py`, from a base 212 commits behind.  Both
predate substantial refactoring; treat as history.

### `2x2_playground`, `benchmark_playground`, `playground`, `refactor`, `unit_tests`

Fully contained in pre-merge master — either identical to it or strict ancestors.
Nothing to merge; they can be deleted once that is confirmed.

---

## A cross-cutting note: `src/tred/plots/graph_effq.py`

Twelve branches modify this file, and it is the single biggest source of apparent
merge conflict in the repository.  Most of that conflict is not real.

`graph_effq.py` is not a library module.  It is a **scratch driver** built around
module-level globals populated from YAML, and each study branch works by
commenting and uncommenting blocks of it — which exports are written, which
diagnostics run, which hardcoded shape is active.  Comparing branches shows the
same lines toggled in different combinations:

- `fr_interp` uncomments the hit export and `write_npz()`
- `test_nd_response` uncomments the current export and the metadata block
- `benchmark_accuracy` comments large export blocks *out*
- `sp` comments out the effq drift-time offset
- `benchmark_output_size` inserts `if True: … continue` to skip readout

Merging these textually yields a driver that performs every study's exports at
once and belongs to no study.  There is no correct textual merge because the
branches encode mutually exclusive configurations of one mutable script.

**The durable fix is to turn these toggles into config keys** — the file already
reads nearly forty settings from YAML, so the mechanism exists and the toggles
simply have not been migrated to it.  Until that is done, the practical rule is:
pick one canonical state for the driver, take from each branch only the changes
that add genuine capability (new config globals, new function calls, new
model dispatch), and leave the on/off toggles alone.

Note also two pre-existing defects in this file, present on master and unrelated
to any branch: an undefined `logger` in the memory-snapshot exception handler,
and an unguarded `torch.cuda.max_memory_allocated()` call that will raise on a
CPU-only run.
