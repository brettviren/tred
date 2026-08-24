'''
Peak-memory and timing profile of raster.steps.eval_qeff().

This is the "before" measurement for tred-02x, whose purpose is to bound peak
memory during rasterization.  It is a REPORT, not a test: wall times and RSS
are not reproducible enough to assert on.  The numerical side of the baseline
is asserted separately by `test_regression_baseline.py` (the `qeff_*` cases).

Two numbers per geometry:

- **analytic** -- the size of the dominant intermediate tensor, computed from
  the charge-box shape and the quadrature order.  Exact and deterministic, and
  the figure the port should be judged against.
- **measured** -- peak RSS of a fresh subprocess that rasterizes only that
  geometry.  A subprocess is required because ru_maxrss is a high-water mark
  for the whole process, so successive measurements in one process would only
  ever report the largest so far.  The parent must also never rasterize
  anything itself: subprocess spawning forks, and the child's ru_maxrss
  inherits the parent's high-water mark, so a parent that had already built a
  big tensor poisons every subsequent child measurement.  All rasterizing
  therefore happens in the children.

Note that `tracemalloc` is useless here: torch CPU tensors are allocated
outside Python's allocator, so tracemalloc reports ~0 for a 400 MB tensor.

Usage:

    python tests/regression/qeff_memory_profile.py             # table
    python tests/regression/qeff_memory_profile.py --json      # machine form
    python tests/regression/qeff_memory_profile.py --case NAME # one, in-process
'''

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time

import torch

from tred.raster.steps import compute_charge_box, compute_qeff, too_short

F64 = torch.float64
GRID = [0.4, 0.4, 0.05]

# name -> (X0, X1, Sigma, Q, n_sigma, npoints)
GEOMETRIES = {
    'point_like': ([[1., 1., 1.]], [[1.001, 1., 1.]], [[0.1, 0.1, 0.05]], [1000.], 3.0, (2, 2, 2)),
    'short_line': ([[1., 1., 1.]], [[1.2, 1.05, 1.05]], [[0.1, 0.1, 0.05]], [1000.], 3.0, (2, 2, 2)),
    'line': ([[1., 1., 1.]], [[3., 2., 1.5]], [[0.1, 0.1, 0.05]], [1000.], 3.0, (2, 2, 2)),
    'wide_diffusion': ([[1., 1., 1.]], [[1.2, 1., 1.]], [[0.5, 0.5, 0.3]], [1000.], 3.0, (2, 2, 2)),
    'very_wide_diffusion': ([[1., 1., 1.]], [[1.2, 1., 1.]], [[2.0, 2.0, 1.0]], [1000.], 3.0, (2, 2, 2)),
    'anisotropic_z': ([[1., 1., 1.]], [[1.05, 1., 6.]], [[0.08, 0.08, 0.05]], [1000.], 3.0, (2, 2, 2)),
    'long_track': ([[0., 0., 0.]], [[20., 12., 8.]], [[0.3, 0.3, 0.2]], [50000.], 3.0, (2, 2, 2)),
    'long_track_diffuse': ([[0., 0., 0.]], [[20., 12., 8.]], [[1.0, 1.0, 0.6]], [50000.], 3.0, (2, 2, 2)),
    'batch_32_steps': (None, None, None, None, 3.0, (2, 2, 2)),   # built below
    'npoints_4': ([[1., 1., 1.]], [[3., 2., 1.5]], [[0.1, 0.1, 0.05]], [1000.], 3.0, (4, 4, 4)),
    'nsigma_6': ([[1., 1., 1.]], [[3., 2., 1.5]], [[0.1, 0.1, 0.05]], [1000.], 6.0, (2, 2, 2)),
}


def _tensors(name):
    '''Build the input tensors for a named geometry.'''
    if name == 'batch_32_steps':
        n = 32
        t = torch.linspace(0.0, 1.0, n, dtype=F64).unsqueeze(1)
        X0 = torch.cat([t * 10, t * 6, t * 4], dim=1)
        X1 = X0 + torch.tensor([[0.6, 0.4, 0.3]], dtype=F64)
        Sigma = torch.full((n, 3), 0.2, dtype=F64)
        Sigma[:, 2] = 0.1
        Q = torch.full((n,), 1000.0, dtype=F64)
        return X0, X1, Sigma, Q, 3.0, (2, 2, 2)
    X0, X1, Sigma, Q, ns, npts = GEOMETRIES[name]
    return (torch.tensor(X0, dtype=F64), torch.tensor(X1, dtype=F64),
            torch.tensor(Sigma, dtype=F64), torch.tensor(Q, dtype=F64), ns, npts)


def analytic(name):
    '''
    Dominant intermediate size, from the charge box and quadrature order.

    eval_qeff() materialises the per-quadrature-node charge field at
    (nbatch, *lmn, *shape) before reducing it.  With the point/line split it
    does so more than once:

        charge  = torch.zeros((nbatch, *lmn, *shape))          # then discarded
        mask    = too_short_mask expanded to the same shape    # bool
        charge  = torch.where(mask, eval_qmodel(qpoint...),    # both evaluated
                                    eval_qmodel(qline...))     # eagerly

    so several float64 tensors of that shape are live at once, plus a bool of
    the same shape.  Eliminating the double evaluation (tred-02x.4) and then
    slicing the box (tred-02x.7) both attack this term.

    Calibration against measured RSS on the larger cases puts the true peak at
    roughly 7-8x `field_bytes`, not the 4x a naive count of the lines above
    suggests -- eval_qmodel() has intermediates of its own.  The estimate below
    keeps the 4x convention because what matters is that it scales with the
    real thing (the measured/estimated ratio is 7.1-8.2 across the big cases,
    i.e. stable), so it works as a cheap deterministic proxy.  Use the measured
    RSS column for absolute figures.
    '''
    X0, X1, Sigma, Q, ns, npts = _tensors(name)
    offset, shape = compute_charge_box(X0, X1, Sigma, (ns, ns, ns), (0, 0, 0),
                                       torch.tensor(GRID, dtype=F64), dtype=F64)
    shape = [int(s) for s in shape]
    nbatch = int(Q.size(0))
    lmn_prod = int(math.prod(npts))
    box_elems = int(math.prod(shape))
    field_elems = nbatch * lmn_prod * box_elems
    return {
        'nbatch': nbatch,
        'box_shape': shape,
        'box_elements': box_elems,
        'lmn_prod': lmn_prod,
        'field_elements': field_elems,
        'field_bytes': field_elems * 8,
        # four float64 copies plus one bool of the same element count
        'estimated_peak_bytes': field_elems * 8 * 4 + field_elems,
        'n_point_like': int(too_short(X0, X1, Sigma, threshold=0.1).sum()),
    }


def run_one(name, repeats=3, xyz_limit=None):
    '''Rasterize one geometry in this process and time it.'''
    X0, X1, Sigma, Q, ns, npts = _tensors(name)
    gs = torch.tensor(GRID, dtype=F64)
    best = float('inf')
    qeff = offset = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        qeff, offset = compute_qeff(Q=Q, X0=X0, X1=X1, Sigma=Sigma,
                                    n_sigma=(ns, ns, ns), origin=(0, 0, 0),
                                    grid_spacing=gs, method='gauss_legendre',
                                    npoints=npts, xyz_limit=xyz_limit)
        best = min(best, time.perf_counter() - t0)
    return qeff, offset, best


def measure_child(name):
    '''Run one geometry in a fresh subprocess; return its RSS, time and result.'''
    here = os.path.abspath(__file__)
    proc = subprocess.run(
        [sys.executable, here, '--case', name],
        capture_output=True, text=True,
        env={**os.environ, 'MPLBACKEND': 'Agg'})
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith('CHILD_JSON='):
            return json.loads(line.split('=', 1)[1])
    return None


def baseline_rss():
    '''RSS of a subprocess that imports torch and does nothing else.'''
    proc = subprocess.run([sys.executable, '-c', 'import torch, resource; '
                           'print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)'],
                          capture_output=True, text=True)
    return int(proc.stdout.strip()) if proc.returncode == 0 else 0


def profile_all():
    '''
    Collect the profile WITHOUT rasterizing in this process.

    analytic() only calls compute_charge_box(), which allocates nothing of
    consequence, so the parent stays small and each child's ru_maxrss is its
    own.
    '''
    base = baseline_rss()
    rows = []
    for name in GEOMETRIES:
        info = analytic(name)
        child = measure_child(name) or {}
        rss = child.get('rss_bytes')
        rows.append({
            'case': name,
            **info,
            'out_shape': child.get('out_shape'),
            'charge_sum': child.get('charge_sum'),
            'seconds': child.get('seconds'),
            'rss_bytes': rss,
            'rss_over_import_bytes': (rss - base) if rss else None,
        })
    return {'torch': torch.__version__, 'grid_spacing': GRID,
            'import_rss_bytes': base, 'rows': rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--case', help='run one geometry and print its peak RSS')
    ap.add_argument('--xyz-limit', type=int, default=None,
                    help='slab limit for the sliced path (tred-02x.7)')
    args = ap.parse_args()

    if args.case:
        qeff, _offset, secs = run_one(args.case, repeats=1,
                                      xyz_limit=args.xyz_limit)
        print('CHILD_JSON=' + json.dumps({
            'rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'seconds': secs,
            'out_shape': list(qeff.shape),
            'charge_sum': float(qeff.sum()),
        }))
        return 0

    report = profile_all()
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f'eval_qeff peak-memory profile   torch {report["torch"]}   '
          f'grid {report["grid_spacing"]}')
    print(f'baseline RSS after importing torch: '
          f'{report["import_rss_bytes"]/1e6:.1f} MB\n')
    hdr = (f'{"case":22} {"box shape":18} {"box el":>9} {"field el":>11} '
           f'{"est peak":>10} {"RSS-base":>10} {"ms":>8} {"sum":>10}')
    print(hdr)
    print('-' * len(hdr))
    for r in report['rows']:
        over = r['rss_over_import_bytes']
        print(f'{r["case"]:22} {str(r["box_shape"]):18} {r["box_elements"]:9d} '
              f'{r["field_elements"]:11d} {r["estimated_peak_bytes"]/1e6:9.2f}M '
              f'{(over/1e6 if over else 0):9.1f}M {r["seconds"]*1e3:8.2f} '
              f'{r["charge_sum"]:10.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
