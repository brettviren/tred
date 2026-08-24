'''
The resolvable-sigma floor.

eval_qeff() integrates the charge model with an npoints-per-cell
Gauss-Legendre rule. When sigma is small compared with a cell the nodes can
miss the peak and the rasterized total is wrong -- silently, and in either
direction. apply_min_sigma() raises sigma to a per-axis floor of half a cell,
where the quadrature error disappears into the box-truncation error that is
present anyway.

See tred-xsj.
'''

import pytest
import torch

from tred.raster import steps
from tred.raster.steps import MIN_SIGMA_CELL_FRACTION, apply_min_sigma, compute_qeff

F64 = torch.float64
GRID = [0.4, 0.4, 0.05]
GRID_T = torch.tensor(GRID, dtype=F64)


def _rasterize(sigma_t, sigma_z=0.05, centre=(1.2, 1.2, 1.0), q=1000.0, **kwargs):
    mid = torch.tensor([centre], dtype=F64)
    qeff, _ = compute_qeff(
        Q=torch.tensor([q], dtype=F64),
        X0=mid, X1=mid + torch.tensor([[1e-4, 0.0, 0.0]], dtype=F64),
        Sigma=torch.tensor([[sigma_t, sigma_t, sigma_z]], dtype=F64),
        n_sigma=(3.0, 3.0, 3.0), origin=(0, 0, 0), grid_spacing=GRID_T,
        method='gauss_legendre', npoints=(2, 2, 2), **kwargs)
    return float(qeff.sum())


# --------------------------------------------------------------------------
# apply_min_sigma itself
# --------------------------------------------------------------------------

def test_auto_floor_is_half_a_cell():
    sigma = torch.zeros((1, 3), dtype=F64)
    out = apply_min_sigma(sigma, GRID_T, 'auto')
    expected = torch.tensor(GRID, dtype=F64) * MIN_SIGMA_CELL_FRACTION
    assert torch.allclose(out[0], expected)


def test_floor_leaves_well_resolved_sigma_alone():
    sigma = torch.tensor([[0.4, 0.4, 0.2]], dtype=F64)
    assert torch.equal(apply_min_sigma(sigma, GRID_T, 'auto'), sigma)


def test_floor_is_per_axis():
    # transverse below the floor, longitudinal above it
    sigma = torch.tensor([[0.01, 0.01, 1.0]], dtype=F64)
    out = apply_min_sigma(sigma, GRID_T, 'auto')
    assert out[0, 0] == pytest.approx(0.2)
    assert out[0, 1] == pytest.approx(0.2)
    assert out[0, 2] == pytest.approx(1.0)      # untouched


def test_floor_can_be_disabled():
    sigma = torch.zeros((1, 3), dtype=F64)
    assert torch.equal(apply_min_sigma(sigma, GRID_T, None), sigma)


def test_explicit_scalar_and_per_axis_floors():
    sigma = torch.zeros((1, 3), dtype=F64)
    assert torch.allclose(apply_min_sigma(sigma, GRID_T, 0.7)[0],
                          torch.full((3,), 0.7, dtype=F64))
    assert torch.allclose(apply_min_sigma(sigma, GRID_T, [0.1, 0.2, 0.3])[0],
                          torch.tensor([0.1, 0.2, 0.3], dtype=F64))


def test_bad_string_rejected():
    with pytest.raises(ValueError):
        apply_min_sigma(torch.zeros((1, 3), dtype=F64), GRID_T, 'sometimes')


def test_warns_once(caplog):
    steps._warned_min_sigma = False
    narrow = torch.zeros((1, 3), dtype=F64)
    with caplog.at_level('WARNING', logger='tred.raster.steps'):
        apply_min_sigma(narrow, GRID_T, 'auto')
        apply_min_sigma(narrow, GRID_T, 'auto')
    assert sum('resolvable floor' in r.message for r in caplog.records) == 1


# --------------------------------------------------------------------------
# the effect it exists for
# --------------------------------------------------------------------------

@pytest.mark.parametrize('sigma_t', [0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.0])
def test_charge_is_conserved_at_every_width(sigma_t):
    '''
    The point of the floor. Before it, sigma_t=0.1 lost 28% of the charge,
    0.05 lost 97.5%, 0.02 returned exactly zero and 0.0 returned NaN.
    '''
    total = _rasterize(sigma_t)
    assert abs(total - 1000.0) / 1000.0 < 0.02, \
        f'sigma_t={sigma_t}: rasterized {total} of 1000'


def test_zero_sigma_no_longer_yields_nan():
    '''
    Compounding case: drift.diffuse() can hand back a zero sigma, and this
    used to turn the whole event to NaN.
    '''
    assert not torch.isnan(torch.tensor(_rasterize(0.0)))


def test_disabling_the_floor_restores_the_bug():
    '''
    Guards against the floor being a no-op that happens to pass.

    How wrong the unfloored result is depends on where the deposit sits within
    a cell -- at sigma_t=0.05 it ranges from about 25 to about 583 of 1000 --
    so assert that it is badly wrong rather than pinning a number.
    '''
    unfloored = _rasterize(0.05, min_sigma=None)
    assert abs(unfloored - 1000.0) / 1000.0 > 0.10, \
        f'expected the unfloored quadrature to be badly wrong, got {unfloored}'
    assert abs(_rasterize(0.05) - 1000.0) / 1000.0 < 0.02


@pytest.mark.parametrize('offset', [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 0.875])
def test_conservation_holds_wherever_the_deposit_sits_in_a_cell(offset):
    '''
    The unfloored error oscillates with sub-cell position, over-estimating as
    well as under-estimating, so a single position would not have caught it.
    '''
    total = _rasterize(0.05, centre=(1.2 + offset * 0.4, 1.2, 1.0))
    assert abs(total - 1000.0) / 1000.0 < 0.02


def test_raster_node_applies_the_floor_by_default():
    from tred.graph import Raster
    assert Raster(velocity=-1.6, grid_spacing=GRID)._min_sigma == 'auto'
    assert Raster(velocity=-1.6, grid_spacing=GRID, min_sigma=None)._min_sigma is None
