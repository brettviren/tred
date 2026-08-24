'''
Slab-at-a-time rasterization bounded by xyz_limit.

eval_qeff() normally materialises the charge field over the whole charge box,
so peak memory is set by the box.  With xyz_limit it evaluates, convolves and
places one slab at a time along the largest over-limit axis.

Each slab is grown by a halo of (stencil-1)/2 cells so the convolution sees its
true neighbours, and zero padding replaces the halo only where it runs off the
box -- which is what padding='same' would have supplied.  The result is
therefore the same computation, not an approximation.

Ported from the idea in the mem_xyzlimit branch; the branch's own code does not
run against master.  See tred-02x.7.
'''

import pytest
import torch

from tred.graph import Drifter, Raster
from tred.raster.steps import _slice_axis, _xyz_limit_vector, compute_qeff

F64 = torch.float64
GRID = torch.tensor([0.4, 0.4, 0.05], dtype=F64)

# Slicing changes the order of accumulation in the convolution, so results
# agree to round-off rather than bit-for-bit.  Measured across these cases the
# worst deviation is under 1e-18 absolute against peak values of order 10, i.e.
# well below one ULP; the charge total is bit-identical.
SLICE_ATOL = 1e-15


def _qeff(X0, X1, Sigma, Q, xyz_limit=None, npoints=(2, 2, 2), n_sigma=3.0):
    return compute_qeff(
        Q=torch.tensor(Q, dtype=F64), X0=torch.tensor(X0, dtype=F64),
        X1=torch.tensor(X1, dtype=F64), Sigma=torch.tensor(Sigma, dtype=F64),
        n_sigma=(n_sigma,) * 3, origin=(0, 0, 0), grid_spacing=GRID,
        method='gauss_legendre', npoints=npoints, xyz_limit=xyz_limit)


CASES = {
    'point': ([[1., 1., 1.]], [[1.001, 1., 1.]], [[0.4, 0.4, 0.2]], [1000.]),
    'line': ([[1., 1., 1.]], [[3., 2., 1.5]], [[0.4, 0.4, 0.2]], [1000.]),
    'anisotropic_z': ([[1., 1., 1.]], [[1.05, 1., 6.]], [[0.4, 0.4, 0.2]], [1000.]),
    'wide': ([[1., 1., 1.]], [[1.2, 1., 1.]], [[0.5, 0.5, 0.3]], [1000.]),
    'batch': ([[1., 1., 1.], [0., 0., 0.]], [[1.001, 1., 1.], [2., 1., .5]],
              [[0.4, 0.4, 0.2], [0.4, 0.4, 0.2]], [1000., 500.]),
}


# --------------------------------------------------------------------------
# equivalence
# --------------------------------------------------------------------------

@pytest.mark.parametrize('case', sorted(CASES))
@pytest.mark.parametrize('limit', [1, 2, 3, 5, 7, 11, 1000])
def test_sliced_matches_unsliced(case, limit):
    '''
    The core claim. Limit 1 is the pathological extreme (one cell per slab,
    all halo) and 1000 exceeds every box here, so slicing never engages.
    '''
    X0, X1, Sigma, Q = CASES[case]
    ref, ref_off = _qeff(X0, X1, Sigma, Q)
    got, got_off = _qeff(X0, X1, Sigma, Q, xyz_limit=limit)

    assert torch.equal(ref_off, got_off), 'offset must not depend on slicing'
    assert ref.shape == got.shape
    assert torch.allclose(ref, got, rtol=0.0, atol=SLICE_ATOL), \
        f'max deviation {float((ref - got).abs().max())}'


@pytest.mark.parametrize('case', sorted(CASES))
def test_charge_total_is_unaffected_by_slicing(case):
    '''Stronger than allclose on the field: the total must not drift at all.'''
    X0, X1, Sigma, Q = CASES[case]
    ref = float(_qeff(X0, X1, Sigma, Q)[0].sum())
    for limit in (1, 3, 8):
        got = float(_qeff(X0, X1, Sigma, Q, xyz_limit=limit)[0].sum())
        assert got == pytest.approx(ref, rel=1e-14, abs=1e-9)


@pytest.mark.parametrize('npoints', [(2, 2, 2), (3, 3, 3), (4, 4, 4)])
def test_slicing_is_independent_of_quadrature_order(npoints):
    '''
    The slab extents must come from the convolution output, never from an
    assumed stencil width -- the mem_xyzlimit branch hardcoded 2, which is
    wrong against master's 3-wide stencil.
    '''
    X0, X1, Sigma, Q = CASES['line']
    ref, _ = _qeff(X0, X1, Sigma, Q, npoints=npoints)
    got, _ = _qeff(X0, X1, Sigma, Q, npoints=npoints, xyz_limit=4)
    assert torch.allclose(ref, got, rtol=0.0, atol=SLICE_ATOL)


def test_uneven_final_slab():
    '''A box length that is not a multiple of the limit must still be exact.'''
    X0, X1, Sigma, Q = CASES['anisotropic_z']
    ref, _ = _qeff(X0, X1, Sigma, Q)
    length = ref.shape[3]
    for limit in (length - 1, length // 2 + 1, length // 3 + 1):
        got, _ = _qeff(X0, X1, Sigma, Q, xyz_limit=limit)
        assert torch.allclose(ref, got, rtol=0.0, atol=SLICE_ATOL), f'limit={limit}'


# --------------------------------------------------------------------------
# axis choice and validation
# --------------------------------------------------------------------------

def test_slices_the_largest_over_limit_axis():
    '''
    Not simply the first. The branch's if/elif cascade always took x when x was
    over the limit, even when another axis was far larger -- the opposite of
    what reduces memory.
    '''
    assert _slice_axis(torch.tensor([7, 7, 125]), 5) == 2
    assert _slice_axis(torch.tensor([125, 7, 7]), 5) == 0
    assert _slice_axis(torch.tensor([7, 125, 7]), 5) == 1
    # only x is over its own limit, so x it is
    assert _slice_axis(torch.tensor([10, 100, 100]), [5, 1000, 1000]) == 0


def test_no_slicing_when_nothing_exceeds_the_limit():
    assert _slice_axis(torch.tensor([7, 7, 7]), 10) is None
    assert _slice_axis(torch.tensor([7, 7, 7]), None) is None


def test_limit_normalisation():
    assert _xyz_limit_vector(torch.tensor([4, 5, 6]), None) is None
    assert _xyz_limit_vector(torch.tensor([4, 5, 6]), 7) == [7, 7, 7]
    assert _xyz_limit_vector(torch.tensor([4, 5, 6]), [1, 2, 3]) == [1, 2, 3]
    assert _xyz_limit_vector(torch.tensor([4, 5, 6]),
                             torch.tensor([2, 3, 4])) == [2, 3, 4]


@pytest.mark.parametrize('bad', [0, -1, [1, 0, 3], [1, 2]])
def test_invalid_limits_are_rejected(bad):
    with pytest.raises(ValueError):
        _xyz_limit_vector(torch.tensor([4, 5, 6]), bad)


# --------------------------------------------------------------------------
# the node
# --------------------------------------------------------------------------

def test_raster_defaults_to_no_slicing():
    assert Raster(velocity=-1.6, grid_spacing=GRID)._xyz_limit is None


def test_raster_forwards_the_limit_without_changing_results():
    drifter = Drifter(diffusion=[0.1, 0.05, 0.05], lifetime=10.0,
                      velocity=-1.6, target=0.0)
    drifted = drifter(torch.tensor([0., 0.]), torch.tensor([1000., 800.]),
                      torch.tensor([[2., 1., 1.], [3., .5, .5]]),
                      torch.tensor([[2.4, 1.2, 1.1], [3.4, .9, .8]]))
    ref = Raster(velocity=-1.6, grid_spacing=GRID)(*drifted)
    for limit in (4, 8, 16):
        got = Raster(velocity=-1.6, grid_spacing=GRID, xyz_limit=limit)(*drifted)
        assert torch.equal(ref.location, got.location)
        assert torch.allclose(ref.data, got.data, rtol=0.0, atol=SLICE_ATOL)
        assert float(got.data.sum()) == pytest.approx(float(ref.data.sum()),
                                                      rel=1e-14)
