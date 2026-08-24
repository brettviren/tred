'''
Numerical regression cases for the drift -> raster -> chunk -> convo chain.

Each case is a function returning a dict of named tensors (and plain numbers).
`test_regression_baseline.py` compares those against fixtures recorded under
`baselines/`, so that a change which is meant to be behaviour-preserving can be
shown to be, and a change which is meant to alter behaviour shows exactly what
it altered.

Cases run on CPU with a fixed seed and are deliberately small.

To record or re-record fixtures:

    python tests/regression/baseline_cases.py --write            # all
    python tests/regression/baseline_cases.py --write diffuse_   # by prefix

Re-recording is a deliberate act.  A diff in the fixture files is the
reviewable evidence of a behaviour change.
'''

import argparse
import json
import os
import sys
import tracemalloc

import torch

from tred.blocking import Block
from tred.chunking import accumulate
from tred.drift import absorb, diffuse
from tred.graph import ChunkSum, Charge, Current, Drifter, LacedConvo, Raster, Sim
from tred.raster.steps import compute_qeff, too_short

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(HERE, 'baselines')

F64 = torch.float64
SEED = 20260824

# Grid used throughout: coarse transverse pitch, fine time bins.  Roughly the
# ND-LAr proportions (pitch ~4mm, tick ~0.5mm of drift) without being tied to
# any particular geometry file.
GRID = [0.4, 0.4, 0.05]

# Charge sits at positive x and drifts to an anode at x=0.  transport() is
# (target - locs)/velocity, so the velocity must be negative for the drift time
# to come out positive.  Getting this sign wrong yields sigma=0 and then a
# silently NaN charge rather than any diagnostic (see tred-xsj).
VELOCITY = -1.6


def _seed():
    torch.manual_seed(SEED)


def _block_summary(block, prefix):
    '''
    Reduce a Block to comparable quantities.

    Block counts matter as much as values here: chunking.accumulate() prunes
    all-zero chunks, and whether it does so is exactly what the end_to_end
    branch changes (tred-4pz.15).
    '''
    return {
        f'{prefix}_location': block.location.to(torch.int64),
        f'{prefix}_data': block.data.to(F64),
        f'{prefix}_nbatches': torch.tensor(block.nbatches, dtype=torch.int64),
        f'{prefix}_sum': block.data.to(F64).sum(),
    }


# --------------------------------------------------------------------------
# drift.diffuse
# --------------------------------------------------------------------------

def diffuse_scalar_diffusion():
    dt = torch.tensor([0.5, 1.0, 2.0, 10.0], dtype=F64)
    return {'sigma': diffuse(dt, 5.0)}


def diffuse_vector_diffusion():
    dt = torch.tensor([0.5, 1.0, 2.0], dtype=F64)
    diffusion = torch.tensor([6.627, 13.243, 13.243], dtype=F64)
    return {'sigma': diffuse(dt, diffusion)}


def diffuse_with_initial_sigma():
    dt = torch.tensor([0.5, 1.0, 2.0], dtype=F64)
    sigma0 = torch.tensor([0.3, 0.3, 0.3], dtype=F64)
    return {'sigma': diffuse(dt, 5.0, sigma0)}


def diffuse_negative_dt():
    '''
    Negative dt means the drift "backed up"; unphysical but reachable.

    sqrt(2*D*dt) goes NaN and master masks it to zero afterwards.  Without an
    initial sigma the masked result and a dt-clamped result agree, so this case
    alone does not distinguish them -- see diffuse_negative_dt_with_sigma.
    '''
    dt = torch.tensor([1.0, -0.5, -5.0], dtype=F64)
    return {'sigma': diffuse(dt, 5.0)}


def diffuse_negative_dt_with_sigma():
    '''
    The case that distinguishes NaN-masking from clamping negative dt.

    Master returns sigma = 0 for the negative-dt entries, discarding the
    initial width and leaving a degenerate zero-width distribution.  Clamping
    dt to zero instead preserves the initial sigma, which is what absorb()'s
    docstring says should happen for negative dt.

    tred-4pz.6 changes this case deliberately.  It also feeds tred-xsj: a
    sigma of zero makes compute_qeff() return NaN.
    '''
    dt = torch.tensor([1.0, -0.5, -5.0], dtype=F64)
    sigma0 = torch.tensor([0.3, 0.3, 0.3], dtype=F64)
    return {'sigma': diffuse(dt, 5.0, sigma0)}


# --------------------------------------------------------------------------
# drift.absorb
# --------------------------------------------------------------------------

def absorb_no_fluctuate():
    dt = torch.tensor([0.0, 1.0, 5.0, 20.0], dtype=F64)
    charge = torch.tensor([1000.0, 1000.0, 1000.0, 1000.0], dtype=F64)
    return {'charge': absorb(charge, dt, 10.0)}


def absorb_fluctuate():
    _seed()
    dt = torch.tensor([0.0, 1.0, 5.0, 20.0], dtype=F64)
    charge = torch.tensor([1000.0, 1000.0, 1000.0, 1000.0], dtype=F64)
    return {'charge': absorb(charge, dt, 10.0, fluctuate=True)}


def absorb_negative_dt():
    dt = torch.tensor([1.0, -0.5, -5.0], dtype=F64)
    charge = torch.tensor([1000.0, 1000.0, 1000.0], dtype=F64)
    return {'charge': absorb(charge, dt, 10.0)}


def absorb_non_integer_charge():
    '''
    absorb() casts charge to int32 before doing anything else, so fractional
    input charge is truncated.  tred-4pz.16 must decide whether to keep that;
    the end_to_end branch removes the cast for differentiability.
    '''
    dt = torch.tensor([1.0, 1.0, 1.0], dtype=F64)
    charge = torch.tensor([100.9, 0.7, 1000.4], dtype=F64)
    return {
        'charge': absorb(charge, dt, 10.0),
        'charge_fluct_input_dtype_is_int': torch.tensor(1, dtype=torch.int64),
    }


# --------------------------------------------------------------------------
# raster: compute_qeff
# --------------------------------------------------------------------------

def _qeff(X0, X1, Sigma, Q, n_sigma=3.0, npoints=(2, 2, 2)):
    X0 = torch.tensor(X0, dtype=F64)
    X1 = torch.tensor(X1, dtype=F64)
    Sigma = torch.tensor(Sigma, dtype=F64)
    Q = torch.tensor(Q, dtype=F64)
    tracemalloc.start()
    qeff, offset = compute_qeff(
        Q=Q, X0=X0, X1=X1, Sigma=Sigma,
        n_sigma=(n_sigma, n_sigma, n_sigma), origin=(0, 0, 0),
        grid_spacing=torch.tensor(GRID, dtype=F64),
        method='gauss_legendre', npoints=npoints)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        'qeff': qeff,
        'offset': offset.to(torch.int64),
        'shape': torch.tensor(list(qeff.shape), dtype=torch.int64),
        'sum': qeff.sum(),
        'too_short': too_short(X0, X1, Sigma, threshold=0.1).to(torch.int64),
        # Recorded for tred-02x, whose whole purpose is to reduce this.  Host
        # allocation via tracemalloc, so it is reproducible without a GPU; it
        # tracks Python-side allocation, not torch's caching allocator, so treat
        # it as an indicator rather than an exact figure.
        '_peak_host_bytes': torch.tensor(peak, dtype=torch.int64),
    }


def qeff_point_like():
    '''Point-like step: takes the qpoint model branch of the too_short split.'''
    return _qeff([[1.0, 1.0, 1.0]], [[1.001, 1.0, 1.0]], [[0.1, 0.1, 0.05]], [1000.0])


def qeff_line():
    '''Extended step: takes the qline model branch.'''
    return _qeff([[1.0, 1.0, 1.0]], [[3.0, 2.0, 1.5]], [[0.1, 0.1, 0.05]], [1000.0])


def qeff_wide_diffusion():
    '''sigma comparable to the grid pitch, where charge is actually conserved.'''
    return _qeff([[1.0, 1.0, 1.0]], [[1.2, 1.0, 1.0]], [[0.5, 0.5, 0.3]], [1000.0])


def qeff_anisotropic():
    '''Long in the drift axis, narrow transversely: the memory-hostile shape.'''
    return _qeff([[1.0, 1.0, 1.0]], [[1.05, 1.0, 6.0]], [[0.08, 0.08, 0.05]], [1000.0])


def qeff_mixed_batch():
    '''Both sides of the too_short split in one batch.'''
    return _qeff([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                 [[1.001, 1.0, 1.0], [2.0, 1.0, 0.5]],
                 [[0.1, 0.1, 0.05], [0.1, 0.1, 0.05]],
                 [1000.0, 500.0])


def qeff_narrow_sigma_charge_loss():
    '''
    Narrow sigma, recorded both with and without the resolvable-sigma floor.

    Without the floor this is tred-xsj in its rawest form: sigma at an eighth
    of a cell, and the quadrature returns 24.8 of 1000.  With the floor (the
    default) it returns 980.

    Both are recorded so the floor cannot be silently removed or its threshold
    changed without this moving.
    '''
    floored = _qeff([[1.0, 1.0, 1.0]], [[1.001, 1.0, 1.0]], [[0.05, 0.05, 0.05]],
                    [1000.0])
    X0 = torch.tensor([[1.0, 1.0, 1.0]], dtype=F64)
    X1 = torch.tensor([[1.001, 1.0, 1.0]], dtype=F64)
    Sigma = torch.tensor([[0.05, 0.05, 0.05]], dtype=F64)
    raw, _ = compute_qeff(
        Q=torch.tensor([1000.0], dtype=F64), X0=X0, X1=X1, Sigma=Sigma,
        n_sigma=(3.0, 3.0, 3.0), origin=(0, 0, 0),
        grid_spacing=torch.tensor(GRID, dtype=F64),
        method='gauss_legendre', npoints=(2, 2, 2), min_sigma=None)
    return {k: v for k, v in floored.items()} | {'unfloored_sum': raw.sum()}


def qeff_min_sigma_floor():
    '''
    The floor engages below half a cell and is inert above it.

    Sweeps sigma across the threshold and records the rasterized total, so a
    change to MIN_SIGMA_CELL_FRACTION shows up here.
    '''
    out = {}
    for tag, sig in (('r100', 0.40), ('r075', 0.30), ('r050', 0.20),
                     ('r025', 0.10), ('r012', 0.05), ('zero', 0.0)):
        res = _qeff([[1.2, 1.2, 1.0]], [[1.2001, 1.2, 1.0]],
                    [[sig, sig, 0.05]], [1000.0])
        out[f'sum_{tag}'] = res['sum']
    return out


def qeff_recenter_batch():
    '''
    Pins what `recenter` actually does, for the tred-02x port.

    `recenter` shifts each step's charge-box offset so a smaller box sits
    centred inside the universal (batch-max) box.  It therefore has NO effect
    on a single-step call, because reduce_to_universal() takes the max over the
    batch and the shift comes out zero -- which is why it needs a batch of
    differing box sizes to be observable at all.

    `skippad` produces results identical to `recenter` because eval_qeff() never
    reads it; see tred-02x.2 and tred-fgh.
    '''
    X0 = torch.tensor([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]], dtype=F64)
    X1 = torch.tensor([[1.6, 1.4, 1.3], [5.1, 5.05, 5.05]], dtype=F64)
    Sigma = torch.tensor([[0.4, 0.4, 0.2], [0.15, 0.15, 0.1]], dtype=F64)
    Q = torch.tensor([1000.0, 1000.0], dtype=F64)
    out = {}
    for label, kwargs in (('default', {}), ('recenter', {'recenter': True}),
                          ('skippad', {'skippad': True})):
        qeff, offset = compute_qeff(
            Q=Q, X0=X0, X1=X1, Sigma=Sigma, n_sigma=(3.0, 3.0, 3.0),
            origin=(0, 0, 0), grid_spacing=torch.tensor(GRID, dtype=F64),
            method='gauss_legendre', npoints=(2, 2, 2), **kwargs)
        out[f'{label}_offset'] = offset.to(torch.int64)
        out[f'{label}_shape'] = torch.tensor(list(qeff.shape), dtype=torch.int64)
        out[f'{label}_sum'] = qeff.sum()
    return out


def qeff_quadrature_orders():
    '''Charge sum against quadrature order, at a ratio where order matters.'''
    out = {}
    for n in (2, 3, 4):
        r = _qeff([[1.0, 1.0, 1.0]], [[1.001, 1.0, 1.0]], [[0.1, 0.1, 0.05]],
                  [1000.0], npoints=(n, n, n))
        out[f'sum_npoints{n}'] = r['sum']
    return out


# --------------------------------------------------------------------------
# graph.Drifter and graph.Raster
# --------------------------------------------------------------------------

def _steps():
    time = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    charge = torch.tensor([1000.0, 800.0, 1200.0], dtype=torch.float32)
    tail = torch.tensor([[2.0, 1.0, 1.0], [3.0, 0.5, 0.5], [1.0, 2.0, 2.0]],
                        dtype=torch.float32)
    head = torch.tensor([[2.4, 1.2, 1.1], [3.0, 0.5, 0.5], [1.8, 2.6, 2.3]],
                        dtype=torch.float32)
    return time, charge, tail, head


def _drifter(fluctuate=False):
    return Drifter(diffusion=[0.1, 0.05, 0.05], lifetime=10.0, velocity=VELOCITY,
                   target=0.0, fluctuate=fluctuate)


def drifter_forward():
    time, charge, tail, head = _steps()
    dsigma, dtime, dcharge, dtail, dhead = _drifter()(time, charge, tail, head)
    return {'dsigma': dsigma.to(F64), 'dtime': dtime.to(F64),
            'dcharge': dcharge.to(F64), 'dtail': dtail.to(F64),
            'dhead': dhead.to(F64)}


def drifter_forward_fluctuate():
    _seed()
    time, charge, tail, head = _steps()
    out = _drifter(fluctuate=True)(time, charge, tail, head)
    return {'dsigma': out[0].to(F64), 'dcharge': out[2].to(F64)}


def raster_forward():
    time, charge, tail, head = _steps()
    drifted = _drifter()(time, charge, tail, head)
    block = Raster(velocity=VELOCITY, grid_spacing=GRID)(*drifted)
    return _block_summary(block, 'qblock')


# --------------------------------------------------------------------------
# chunking.accumulate
# --------------------------------------------------------------------------

def accumulate_overlapping():
    loc = torch.tensor([[0, 0], [0, 0], [2, 2], [2, 2], [4, 4]])
    dat = torch.arange(5 * 2 * 2, dtype=F64).reshape(5, 2, 2)
    return _block_summary(accumulate(Block(location=loc, data=dat)), 'acc')


def accumulate_all_zero():
    '''
    Every chunk is zero.  Master prunes them and returns an EMPTY block; the
    end_to_end branch removes that pruning and returns all of them.

    This is the case tred-4pz.15 turns on, and the reason the pruning must
    become conditional rather than deleted.
    '''
    loc = torch.tensor([[0, 0], [0, 0], [2, 2]])
    dat = torch.zeros((3, 2, 2), dtype=F64)
    return _block_summary(accumulate(Block(location=loc, data=dat)), 'acc')


def accumulate_mixed_zero():
    '''Mixed zero and non-zero: measures how much pruning actually removes.'''
    loc = torch.tensor([[0, 0], [0, 0], [2, 2], [4, 4], [6, 6]])
    dat = torch.zeros((5, 2, 2), dtype=F64)
    dat[2] = 1.0
    dat[4] = 2.5
    return _block_summary(accumulate(Block(location=loc, data=dat)), 'acc')


def accumulate_below_prune_threshold():
    '''
    Content that is non-zero but under the 1e-3 pruning threshold.  Pruning
    discards real (if tiny) charge here, which is worth having pinned before
    the threshold is made configurable.
    '''
    loc = torch.tensor([[0, 0], [2, 2]])
    dat = torch.full((2, 2, 2), 1e-6, dtype=F64)
    return _block_summary(accumulate(Block(location=loc, data=dat)), 'acc')


# --------------------------------------------------------------------------
# graph.ChunkSum, every method
# --------------------------------------------------------------------------

def _chunksum_block():
    loc = torch.tensor([[1, 1], [2, 3], [9, 9]])
    dat = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
    return Block(location=loc, data=dat)


def chunksum_methods():
    '''
    All four methods on identical input.  They legitimately differ in how many
    chunks they emit, so record counts as well as totals: a change in either is
    a behaviour change.
    '''
    out = {}
    for method in ('chunksum', 'chunksum2', 'chunksum_inplace_v1',
                   'chunksum_inplace_v2'):
        block = ChunkSum([4, 4], method=method)(_chunksum_block())
        out[f'{method}_nbatches'] = torch.tensor(block.nbatches, dtype=torch.int64)
        out[f'{method}_sum'] = block.data.to(F64).sum()
        out[f'{method}_location'] = block.location.to(torch.int64)
    return out


# --------------------------------------------------------------------------
# end-to-end Sim
# --------------------------------------------------------------------------

def sim_end_to_end():
    '''
    Short full-chain run: drift -> raster -> chunksum -> convo -> chunksum.

    The response is synthetic and small, so this stays fast and needs no data
    file; it exercises the wiring and the laced convolution, not the physics.
    '''
    _seed()
    nimper = 2
    lacing = torch.tensor([nimper, nimper, 1])

    drifter = _drifter()
    raster = Raster(velocity=VELOCITY, grid_spacing=GRID)
    chunksum_q = ChunkSum((4 * nimper, 4 * nimper, 16))
    convo = LacedConvo(lacing, o_shape=(8, 8, 40))
    chunksum_i = ChunkSum((4, 4, 8), method='chunksum_inplace_v2')

    sim = Sim(Charge(drifter, raster, chunksum_q), Current(convo, chunksum_i))

    # Separable, decaying synthetic response.  The transverse extent must be an
    # odd multiple of the lacing: interlaced_symm_v2() requires the de-laced
    # response length to be odd, so 3*nimper works where 4*nimper does not.
    ax = torch.linspace(0.0, 1.0, 3 * nimper, dtype=torch.float32)
    at = torch.linspace(0.0, 1.0, 24, dtype=torch.float32)
    response = (torch.exp(-3 * ax)[:, None, None]
                * torch.exp(-3 * ax)[None, :, None]
                * torch.exp(-2 * at)[None, None, :])

    time, charge, tail, head = _steps()
    block = sim(response, time, charge, tail, head)
    return _block_summary(block, 'iblock')


# --------------------------------------------------------------------------

CASES = {name: obj for name, obj in sorted(globals().items())
         if callable(obj) and not name.startswith('_')
         and getattr(obj, '__module__', None) == __name__
         and name not in ('main',)}


def run_case(name):
    torch.manual_seed(SEED)
    return CASES[name]()


def write_baselines(prefix=''):
    os.makedirs(BASELINE_DIR, exist_ok=True)
    written = []
    for name in sorted(CASES):
        if not name.startswith(prefix):
            continue
        result = run_case(name)
        path = os.path.join(BASELINE_DIR, f'{name}.pt')
        torch.save({k: torch.as_tensor(v) for k, v in result.items()}, path)
        written.append(name)
    # A prefixed write only re-records some cases, so the recorded case list is
    # everything that now has a fixture on disk -- not just what this call
    # touched.  Deriving it from the directory keeps a partial re-record from
    # silently truncating the list.
    on_disk = sorted(f[:-3] for f in os.listdir(BASELINE_DIR) if f.endswith('.pt'))
    meta = {
        'torch': torch.__version__,
        'seed': SEED,
        'grid_spacing': GRID,
        'cases': on_disk,
    }
    with open(os.path.join(BASELINE_DIR, 'metadata.json'), 'w') as fp:
        json.dump(meta, fp, indent=2, sort_keys=True)
        fp.write('\n')
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write', action='store_true',
                    help='record fixtures instead of printing a summary')
    ap.add_argument('prefix', nargs='?', default='',
                    help='only handle cases whose name starts with this')
    args = ap.parse_args()

    if args.write:
        for name in write_baselines(args.prefix):
            print(f'wrote {name}')
        return 0

    for name in sorted(CASES):
        if not name.startswith(args.prefix):
            continue
        result = run_case(name)
        bits = []
        for key, val in result.items():
            val = torch.as_tensor(val)
            bits.append(f'{key}={val.item():.6g}' if val.numel() == 1
                        else f'{key}:{tuple(val.shape)}')
        print(f'{name}\n    ' + '  '.join(bits))
    return 0


if __name__ == '__main__':
    sys.exit(main())
