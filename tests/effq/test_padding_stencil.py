'''
The interpolation stencil is 3 wide, and eval_qeff() relies on that.

`eval_qeff()` convolves with `padding='same'`, which is symmetric only for an
odd kernel.  It carried the comment

    # FIXME: the padding is only valid for interpolation points == 3

with nothing testing the condition.  These tests establish that the stencil is
3 wide by construction for every supported configuration, and that charge lands
where it should under `padding='same'`.

The width is NOT the quadrature order.  `npoints` sets the number of
Gauss-Legendre nodes per cell and appears as the kernel's first three
dimensions (lmn); the stencil is the last three (rst) and is fixed at 3.
Conflating the two is easy and consequential: the mem_xyzlimit branch hardcoded
a stencil of 2, correct against its own base where `_create_u1d_GL()` returned
a [npt, 2] tensor, and wrong against master where it returns [npt, 3].

See tred-02x.3.
'''

import pytest
import torch

from tred.raster.steps import (compute_qeff, create_wu_block, _create_u1d_GL,
                               _create_u_block)

F64 = torch.float64
GRID = torch.tensor([0.4, 0.4, 0.05], dtype=F64)

STENCIL = 3


@pytest.mark.parametrize('npoints', [(1, 1, 1), (2, 2, 2), (3, 3, 3),
                                     (4, 4, 4), (5, 5, 5), (2, 3, 4)])
def test_stencil_is_three_wide_for_every_quadrature_order(npoints):
    kernel = create_wu_block('gauss_legendre', npoints, GRID, 'cpu', dtype=F64)
    lmn, rst = tuple(kernel.size()[:3]), tuple(kernel.size()[3:])
    assert lmn == tuple(npoints), 'quadrature nodes should be the leading dims'
    assert rst == (STENCIL,) * 3, (
        f'interpolation stencil is {rst}, not {(STENCIL,)*3}; '
        f"eval_qeff()'s padding='same' assumes an odd stencil")


@pytest.mark.parametrize('npt', [1, 2, 3, 4, 5, 6])
def test_u1d_last_dimension_is_the_stencil(npt):
    u1d = _create_u1d_GL(npt, dtype=F64)
    assert u1d.shape == (npt, STENCIL)


def test_u_block_rejects_a_different_stencil():
    '''The width is enforced, not merely conventional.'''
    wrong = torch.zeros((2, 2), dtype=F64)      # stencil of 2, as the old code had
    with pytest.raises(ValueError):
        _create_u_block([wrong, wrong, wrong])


def test_only_gauss_legendre_is_supported():
    # Anything else raises, so the stencil width cannot vary by method either.
    with pytest.raises(NotImplementedError):
        create_wu_block('simpson', (2, 2, 2), GRID, 'cpu', dtype=F64)


def test_same_padding_is_symmetric_for_an_odd_stencil():
    '''
    An odd kernel pads (k-1)/2 on both sides, so a symmetric input stays
    symmetric.  An even kernel would pad asymmetrically and shift the result by
    half a cell.
    '''
    assert STENCIL % 2 == 1
    signal = torch.zeros((1, 1, 7, 7, 7), dtype=F64)
    signal[0, 0, 3, 3, 3] = 1.0
    kernel = torch.ones((1, 1, STENCIL, STENCIL, STENCIL), dtype=F64)
    out = torch.nn.functional.conv3d(signal, kernel, padding='same')[0, 0]
    assert torch.allclose(out, torch.flip(out, dims=(0, 1, 2)))


def _centroid(qeff, offset, batch=0):
    '''
    Charge-weighted centroid in absolute coordinates.

    `offset` indexes the LOWER CORNER of a cell, so the cell centre is at
    (index + offset + 0.5) * grid_spacing.  Omitting the half-cell makes a
    correctly placed raster look like it has a systematic half-cell bias.
    '''
    grid = qeff[batch]
    total = grid.sum()
    out = []
    for axis in range(3):
        others = tuple(j for j in range(3) if j != axis)
        idx = torch.arange(grid.shape[axis], dtype=F64)
        mean_index = (grid.sum(dim=others) * idx).sum() / total
        out.append(float((mean_index + offset[batch, axis] + 0.5) * GRID[axis]))
    return out


@pytest.mark.parametrize('centre', [
    (1.0, 1.0, 1.0),        # on a cell corner
    (1.2, 1.2, 1.025),      # mid-cell in x/y
    (1.37, 0.83, 1.117),    # nothing aligned
])
def test_charge_lands_at_the_step_midpoint(centre):
    '''
    A short, well-resolved step should rasterize with its centroid at its
    midpoint.  sigma is kept comparable to the grid pitch so that tred-xsj's
    charge loss does not distort the centroid.
    '''
    half = torch.tensor([0.05, 0.05, 0.02], dtype=F64)
    mid = torch.tensor(centre, dtype=F64)
    X0, X1 = (mid - half).unsqueeze(0), (mid + half).unsqueeze(0)
    Sigma = torch.tensor([[0.4, 0.4, 0.2]], dtype=F64)
    qeff, offset = compute_qeff(
        Q=torch.tensor([1000.0], dtype=F64), X0=X0, X1=X1, Sigma=Sigma,
        n_sigma=(4.0, 4.0, 4.0), origin=(0, 0, 0), grid_spacing=GRID,
        method='gauss_legendre', npoints=(2, 2, 2))

    got = _centroid(qeff, offset)
    for axis in range(3):
        # tolerance is a tenth of a cell: enough to catch a half-cell shift
        assert abs(got[axis] - float(mid[axis])) < 0.1 * float(GRID[axis]), (
            f'axis {axis}: centroid {got[axis]:.5f} vs midpoint '
            f'{float(mid[axis]):.5f}')


def test_no_half_cell_shift_between_quadrature_orders():
    '''
    Changing the quadrature order must not move the charge, only refine it.
    A stencil/padding mismatch would show up as an order-dependent shift.
    '''
    X0 = torch.tensor([[1.17, 0.93, 1.041]], dtype=F64)
    X1 = torch.tensor([[1.27, 1.03, 1.081]], dtype=F64)
    Sigma = torch.tensor([[0.4, 0.4, 0.2]], dtype=F64)
    centroids = []
    for n in (2, 3, 4):
        qeff, offset = compute_qeff(
            Q=torch.tensor([1000.0], dtype=F64), X0=X0, X1=X1, Sigma=Sigma,
            n_sigma=(4.0, 4.0, 4.0), origin=(0, 0, 0), grid_spacing=GRID,
            method='gauss_legendre', npoints=(n, n, n))
        centroids.append(_centroid(qeff, offset))
    for axis in range(3):
        spread = max(c[axis] for c in centroids) - min(c[axis] for c in centroids)
        assert spread < 0.05 * float(GRID[axis]), \
            f'axis {axis}: centroid moves {spread} across quadrature orders'
