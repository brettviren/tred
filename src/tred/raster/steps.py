import torch
from torch import Tensor

from ..types import index_dtype, MAX_INDEX, MIN_INDEX
# from ..utils import to_tensor

import logging
import os

logger = logging.getLogger('tred.raster.steps')


# Optional torch.compile of the charge-model kernels.
#
# Those kernels are long elementwise expressions over broadcast tensors, and
# peak rasterization memory is set by their intermediates rather than by the
# result.  Letting inductor fuse the chain removes most of those intermediates.
# Measured on CPU, float64, box 66x46x233 (tred-02x.12):
#
#     variant                 peak tensors    time
#     @torch.jit.script            5.0        98 ms      <- default
#     torch.compile(dynamic=True)  2.0        57 ms
#
# "peak tensors" is the slope of peak RSS against result size, i.e. how many
# full-size tensors are live at once, so it is the multiplier that matters for
# a big step.
#
# Off by default.  It costs a one-off ~5 s compile on first call, and inductor
# is not available or reliable in every environment.  Enable with the
# TRED_COMPILE_RASTER environment variable or set_compiled_kernels().
#
# dynamic=True is essential: with dynamic=False every new charge-box shape
# triggers a fresh 2.6-4.2 s compile, and tred's box shapes vary per step.
_COMPILE_KERNELS = os.environ.get('TRED_COMPILE_RASTER', '').lower() \
    in ('1', 'true', 'yes', 'on')
_compiled_kernels = {}


def set_compiled_kernels(enable=True):
    '''
    Turn torch.compile of the charge-model kernels on or off.

    Returns the previous setting.  Compilation happens lazily on the next call
    and is cached, so toggling is cheap.
    '''
    global _COMPILE_KERNELS
    was, _COMPILE_KERNELS = _COMPILE_KERNELS, bool(enable)
    return was


def _kernel(fn):
    '''
    Return fn, or a compiled version of it when compiled kernels are enabled.

    Falls back to the plain function, with a warning, if compilation fails --
    this is an optimization, never a correctness requirement.
    '''
    if not _COMPILE_KERNELS:
        return fn
    if fn not in _compiled_kernels:
        try:
            _compiled_kernels[fn] = torch.compile(fn, dynamic=True)
        except Exception as err:      # inductor unavailable, unsupported, ...
            logger.warning(f'torch.compile of {getattr(fn, "__name__", fn)} '
                           f'failed, using the uncompiled kernel: {err}')
            _compiled_kernels[fn] = fn
    return _compiled_kernels[fn]

DEFAULT_FLOAT_DTYPE = torch.float64


def to_tensor(source, device, dtype=DEFAULT_FLOAT_DTYPE):
    '''Aliasing or create a tensor if not existing.
    Result tensor will be moved to the device

    Gradient-transparent: a tensor arriving with grad tracking keeps it.  This
    used to force requires_grad=False on the way through, which silently
    detached a leaf and raised outright on a non-leaf ("you can only change
    requires_grad flags of leaf variables") -- so the raster could not be used
    inside a gradient computation at all.  Inputs that do not track gradients
    are unaffected, since they already carry requires_grad=False.
    '''
    if isinstance(source, torch.Tensor):
        t = source.to(dtype)
        t = t.to(device)
    else:
        t = torch.tensor(source, dtype=dtype, requires_grad=False,
                         device=device)
    return t


def _coarsest_float_dtype(default_dtype, *values):
    """Return the lowest-precision supported floating dtype among the inputs."""
    rank = {
        torch.float32: 0,
        torch.float64: 1,
    }
    dtypes = []
    if default_dtype in rank:
        dtypes.append(default_dtype)
    elif default_dtype is not None:
        raise TypeError(f'Unsupported floating dtype: {default_dtype}')
    for value in values:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if value.dtype not in rank:
                raise TypeError(f'Unsupported floating dtype: {value.dtype}')
            dtypes.append(value.dtype)
    if not dtypes:
        return DEFAULT_FLOAT_DTYPE
    return min(dtypes, key=rank.__getitem__)


def _snap_near_integer(values: Tensor, source_dtype):
    """Snap values close to an integer boundary before flooring."""
    nearest = torch.round(values)
    eps = torch.finfo(source_dtype).eps
    tol = 8 * eps * torch.clamp(nearest.abs(), min=1.0)
    return torch.where((values - nearest).abs() <= tol, nearest, values)


def compute_coordinate(idxs: Tensor, origin, grid_spacing, device='cpu',
                       dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Arguments:
        idxs : (N, vdim); index of grid point, Tensor of index_dtype
        origin : (vdim,), grid origin
        grid_spacing : (vdim, ), grid spacing
    return
        origin + spacing * idx
    '''
    fidxs = to_tensor(idxs, device=device, dtype=dtype)
    assert torch.all(fidxs <= MAX_INDEX), 'Overflow of index_dtype'
    assert torch.all(fidxs >= MIN_INDEX), 'Underflow of index_dtype'
    idxs = to_tensor(idxs, device=device, dtype=index_dtype)

    if idxs.dim() == 1:
        idxs = idxs.unsqueeze(0)
    origin = to_tensor(origin, device=idxs.device, dtype=dtype)
    grid_spacing = to_tensor(grid_spacing, device=device, dtype=dtype)
    return origin.unsqueeze(0) + idxs * grid_spacing.unsqueeze(0)


def compute_index(coords, origin, grid_spacing, device='cpu',
                  dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Arguments:
        coords : (N, vdim)
        origin : (vdim,)
        grid_spacing : (vdim, )

    return
        (coords - origin)//spacing, index of grid point
    '''
    source_dtype = _coarsest_float_dtype(dtype, coords, origin, grid_spacing)
    coords = to_tensor(coords, device, dtype=torch.float64)
    if coords.dim() == 1:
        coords = coords.unsqueeze(0)
    origin = to_tensor(origin, device, dtype=torch.float64)
    grid_spacing = to_tensor(grid_spacing, device, dtype=torch.float64)
    idxs = (coords - origin.unsqueeze(0)) / grid_spacing.unsqueeze(0)
    idxs = _snap_near_integer(idxs, source_dtype)

    assert torch.all(idxs <= MAX_INDEX), f'Overflow of index_dtype {MAX_INDEX}'
    assert torch.all(idxs >= MIN_INDEX), 'Underflow of index_dtype'
    return idxs.floor().to(index_dtype)


def compute_bounds_X0X1(X0X1, Sigma, n_sigma, dtype=DEFAULT_FLOAT_DTYPE):
    '''
        X0X1 : (N, vdim, 2)
        Sigma : (N, vdim)
        n_sigma: (vdim, )
    return :
        bounds: (N, vdim, 2)
        '''
    X0X1 = to_tensor(X0X1, device=Sigma.device, dtype=dtype)
    Sigma = to_tensor(Sigma, device=Sigma.device, dtype=dtype)
    n_sigma = to_tensor(n_sigma, dtype=dtype, device=Sigma.device)
    offset = (n_sigma.unsqueeze(0) * Sigma) # (N, vdim)
    min_limits = torch.min(X0X1, dim=2).values - offset # torch.min(shape(N,vdim,2)) --> shape(N, vdim)
    max_limits = torch.max(X0X1, dim=2).values + offset
    return torch.stack([min_limits, max_limits], dim=2)


def _stack_X0X1(X0, X1):
    '''
    Arguments:
        X0: (N, vdim)
        X1: (N, vdim)
    Return:
       X0X1: (N, vdim, 2)
     '''
    return torch.stack((X0, X1), dim=2)


def compute_bounds_X0_X1(X0, X1, Sigma, n_sigma, dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Arguments:
        X0: (N, vdim)
        X1: (N, vdim)
        Sigma: (N, vdim)
        n_sigma: (vdim, )
    return:
        (N, vdim, 2) float
    '''
    X0 = to_tensor(X0, device=Sigma.device, dtype=dtype)
    X1 = to_tensor(X1, device=Sigma.device, dtype=dtype)
    Sigma = to_tensor(Sigma, device=Sigma.device, dtype=dtype)
    n_sigma = to_tensor(n_sigma, dtype=dtype, device=Sigma.device)
    combined = _stack_X0X1(X0, X1)
    bounds = compute_bounds_X0X1(combined, Sigma, n_sigma, dtype=dtype)
    return bounds

def reduce_to_universal(shape: Tensor):
    """
    Compute a universal shape across all steps using reductions.

    Args:
        shapes (torch.Tensor): Shapes of the charge boxes (N, vdim).

    Returns:
        Universal shape. (vdim, )
    """
    universal_max = torch.max(shape, dim=0).values

    return universal_max


def compute_charge_box(X0: Tensor, X1: Tensor, Sigma: Tensor,
                       n_sigma, origin, grid_spacing,
                       dtype=DEFAULT_FLOAT_DTYPE, **kwargs):
    '''
    FIXME: Output is only meaningful for vdim == 3.

    Args:
        X0 : (N, vdim), Tensor
        X1 : (N, vdim), Tensor
        Sigma : (N, vdim)
        n_sigma : (vdim, )
        origin : (vdim, )
        grid_spacing : (vdim, )
    Returns:
        offset : (N, vdim)
        shape : (vdim,)
        '''
    recenter = kwargs.get('recenter', False)
    compare_key = kwargs.get('compare_key', 'index')

    device = X0.device
    X0 = to_tensor(X0, device=device, dtype=dtype)
    X1 = to_tensor(X1, device=device, dtype=dtype)
    Sigma = to_tensor(Sigma, device=device, dtype=dtype)

    n_sigma = to_tensor(n_sigma, device, dtype=dtype)

    extremes = compute_bounds_X0_X1(X0, X1, Sigma, n_sigma, dtype=dtype) # (N, vdim, 2)

    min_limit = extremes[:, :, 0] # (N, vdim)
    max_limit = extremes[:, :, 1] # (N, vdim)

    if compare_key == 'index':
        min_limit = compute_index(min_limit, origin, grid_spacing, device=device, dtype=dtype)
        max_limit = compute_index(max_limit, origin, grid_spacing, device=device, dtype=dtype)
        offset = min_limit

        shape = max_limit - min_limit + 1
        universal_shape = reduce_to_universal(shape)

        if recenter:
            # center moved to half the universal shape,
            # so the lower corner shifts by center - half_shape, which are negative values.
            # The new lower corner is smaller than old lower corner by |center - half_shape|
            # offset = offset + (center - half_shape)
            shift = (
                (shape - universal_shape)/2
            ).floor().to(index_dtype)
            offset = offset + shift
            assert torch.all(shift <= 0)

        universal_shape = universal_shape.to(index_dtype)

    elif compare_key == 'coordinate':
        raise NotImplementedError('Not support comparation by coordinate')
        '''
        The index computation cast the coordinate to lower bound.
        I am not sure how to reconcile cases below:
        1) Minimum close to lower bound and maximum close to upper bound.
        2) Minimum close to upper bound and maximum close to lower bound.
        The shape given by the two are off by one.
        '''
    else:
        raise NotImplementedError('Only support comparation by index and coordinate')
    return offset, universal_shape


def qline_diff3D(Q, X0, X1, Sigma, x, y, z):
    """
    Args:
        Q (N,)
        X0 (N, 3)
        X1 (N, 3)
        Sigma (N, 3)
        x (N, other shape)
        y (N, other shape)
        z (N, other shape)
    return:
        q (N, othter shape)
    """
    num_dims_to_add = x.ndim - 1
    shape_new = [-1] + num_dims_to_add * [1] # TorchScript does NOT support (-1,) + dynamic tuples

    # Prepare for broadcasting
    x0, y0, z0 = tuple(X0[:,i].view(shape_new) for i in range(3))
    x1, y1, z1 = tuple(X1[:,i].view(shape_new) for i in range(3))
    sx, sy, sz = tuple(Sigma[:,i].view(shape_new) for i in range(3))
    Q = Q.view(shape_new)
    args = (Q, x0, y0, z0, x1, y1, z1, sx, sy, sz, x, y, z)
    return _kernel(qline_diff3D_script)(*args)


@torch.jit.script
def qline_diff3D_script(
        Q: Tensor, x0: Tensor, y0: Tensor, z0: Tensor,
        x1: Tensor, y1: Tensor, z1: Tensor,
        sx: Tensor, sy: Tensor, sz: Tensor,
        x, y, z
) -> Tensor:
    '''
    Args:
        Q: (N, 1, 1, 1, ...)
        x0, y0, z0, x1, y1, z1, sx, sy, sz: (N, 1, 1, 1, ...)
        x (N, other shape)
        y (N, other shape)
        z (N, other shape)
    return:
        q (N, othter shape)
    '''
    sqrt2 = 1.4142135623730951

    # Calculate differences
    dx01 = x0 - x1
    dy01 = y0 - y1
    dz01 = z0 - z1

    # Calculate squared terms
    sxsy2 = (sx*sy)**2
    sxsz2 = (sx*sz)**2
    sysz2 = (sy*sz)**2
    sx2 = sx**2
    sy2 = sy**2
    sz2 = sz**2

    # Calculate delta terms
    deltaSquare = (
        sysz2 * dx01**2 +
        sxsy2 * dz01**2 +
        sxsz2 * dy01**2
    )
    deltaSquareSqrt = torch.sqrt(deltaSquare)

    # Calculate charge distribution
    QoverDeltaSquareSqrt4pi = Q / (deltaSquareSqrt * 4 * torch.pi)
    erfArgDenominator = sqrt2 * deltaSquareSqrt * sx * sy * sz

    charge = ((-QoverDeltaSquareSqrt4pi * torch.exp(
        -sy2 * torch.pow(x * dz01 + (z1*x0 - z0*x1) - z * dx01, 2) *0.5/deltaSquare ))
              * torch.exp(
        -sx2 * torch.pow(y * dz01 + (z1*y0 - z0*y1) - z * dy01, 2) *0.5/deltaSquare)
              * torch.exp(
        -sz2 * torch.pow(y * dx01 + (x1*y0 - x0*y1) - x * dy01, 2) *0.5/deltaSquare)) * (
        torch.erf((
            sysz2 * (x - x0) * dx01 +
            sxsy2 * (z - z0) * dz01 +
            sxsz2 * (y - y0) * dy01
        )/erfArgDenominator) -
        torch.erf((
            sysz2 * (x - x1) * dx01 +
            sxsy2 * (z - z1) * dz01 +
            sxsz2 * (y - y1) * dy01
        )/erfArgDenominator)
    )
    return charge


def qpoint_diff3D(Q, X0, X1, Sigma, x, y, z):
    """
      Args:
          Q (N,)
          X0 (N, 3)
          X1 (N, 3)
          Sigma (N, 3)
          x (N, other shape)
          y (N, other shape)
          z (N, other shape)
      return:
          q (N, othter shape)
      """
    num_dims_to_add = x.ndim - 1
    shape_new = [-1] + num_dims_to_add * [1]

    # Prepare for broadcasting
    x0, y0, z0 = tuple(X0[:, i].view(shape_new) for i in range(3))
    x1, y1, z1 = tuple(X1[:, i].view(shape_new) for i in range(3))
    sx, sy, sz = tuple(Sigma[:, i].view(shape_new) for i in range(3))
    Q = Q.view(shape_new)
    return _kernel(qpoint_diff3D_kernel)(Q, x0, y0, z0, x1, y1, z1, sx, sy, sz,
                                        x, y, z)


def qpoint_diff3D_kernel(Q, x0, y0, z0, x1, y1, z1, sx, sy, sz, x, y, z):
    '''
    The point-charge expression, split out so it can be routed through
    _kernel() alongside qline_diff3D_script.  See tred-02x.12.
    '''
    xc, yc, zc = (x0+x1)/2, (y0+y1)/2, (z0+z1)/2
    charge = Q / torch.pow(torch.tensor(2 * 3.1415926), 1.5) / sx / sy / sz \
        * torch.exp(- (x - xc)**2 / 2 / sx**2
                    - (y - yc)**2 / 2 / sy**2
                    - (z - zc)**2 / 2 / sz**2)
    return charge


def too_short(X0, X1, Sigma, threshold=0.05):
    #    sqrt2 = 1.4142135623730951
    x0, y0, z0 = [X0[:, i] for i in range(3)]
    x1, y1, z1 = [X1[:, i] for i in range(3)]
    sx, sy, sz = [Sigma[:, i] for i in range(3)]

    # Calculate differences
    dx01 = x0 - x1
    dy01 = y0 - y1
    dz01 = z0 - z1

    point_like = torch.stack(
        [torch.abs(dx01) < threshold * sx,
         torch.abs(dy01) < threshold * sy,
         torch.abs(dz01) < threshold * sz], dim=1
    )
    point_like = torch.all(point_like, dim=1)

    return point_like


try:
    import scipy
    roots_legendre = scipy.special.roots_legendre
except:
    def roots_legendre(n: int) -> tuple[tuple[float]]:
        '''
        A lookup table for GL rule.
        generated by the following
        import scipy
        for i in range(1,7):
            roots, weights = scipy.special.roots_legendre(i)
            roots_str = '(' + ', '.join(f'{x:.16f}' for x in roots) + ')'
            weights_str = '(' + ', '.join(f'{x:.16f}' for x in weights) + ')'
            print(f'if n == {i}:\n    return {roots_str}, {weights_str}')
        '''
        if n == 1:
            return (0.0000000000000000,), (2.0000000000000000,)
        if n == 2:
            return (-0.5773502691896258, 0.5773502691896258), \
                (1.0000000000000000, 1.0000000000000000)
        if n == 3:
            return (-0.7745966692414834, 0.0000000000000000, 0.7745966692414834), \
                (0.5555555555555558, 0.8888888888888883, 0.5555555555555558)
        if n == 4:
            return (-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526), \
                (0.3478548451374536, 0.6521451548625464, 0.6521451548625464, 0.3478548451374536)
        if n == 5:
            return (-0.9061798459386640, -0.5384693101056831, 0.0000000000000000, 0.5384693101056831, 0.9061798459386640), \
                (0.2369268850561896, 0.4786286704993662, 0.5688888888888886, 0.4786286704993662, 0.2369268850561896)
        if n == 6:
            return (-0.9324695142031521, -0.6612093864662645, -0.2386191860831969, 0.2386191860831969, 0.6612093864662645, 0.9324695142031521), \
                (0.1713244923791695, 0.3607615730481392, 0.4679139345726914, 0.4679139345726914, 0.3607615730481392, 0.1713244923791695)
        raise ValueError(f"customized roots_legendre does not support n = {n}"\
                     " (must be 1 <= n <= 6)")


def _create_w1d_GL(npt, spacing, device='cpu', dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Args:
        npt : integer
        spacing : float
    Return:
        a tensor of weights of n-point GL quadrature after correcting for length of intervals
    '''
    _, weights = roots_legendre(npt)
    w1d = torch.tensor(weights, dtype=dtype, requires_grad=False, device=device) * spacing/2
    return w1d

def _create_w1ds(method, npoints, grid_spacing, device='cpu',
                 dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Args:
        method : str
        npoints : (vdim, ), integers
        grid_spacing : (vdim, ), float
    return
        (vdim, ), a list in which each element is a tensor of weights of
        n-point GL quadrature after correcting for length of intervals
    '''
    if method != 'gauss_legendre':
        raise NotImplementedError('Not implemented method but gauss legendre quadrature')
    w1ds = []
    for ipt, npt in enumerate(npoints):
        w1ds.append(_create_w1d_GL(npt, grid_spacing[ipt], device, dtype=dtype))
    return w1ds

def _create_w_block(w1ds):
    '''create a weight block
    Args:
        w1ds: tuple[Tensor, Tensor, ...], (Tensor_1, Tensor_2, ...), length of tuple is vdim,
    Return:
        weights: Tensor, in a shape of (N_1, N_2, ..., N_i, ...) for i from 1 to vdim
    '''
    ndim = len(w1ds)
    shape_0 = [-1,] + (ndim-1) * [1,] # special case [-1] + 0 * [1] --> [-1]
    wblock = w1ds[0].view(shape_0)
    for i in range(1, ndim, 1):
        shape_new = [1,] * ndim
        shape_new[i] = -1
        wblock = wblock * w1ds[i].view(shape_new)
    return wblock

def create_w_block(method, npoints, grid_spacing, device='cpu',
                   dtype=DEFAULT_FLOAT_DTYPE):
    '''create a weight block
    Args:
        method : str
        npoints : tuple[int], (vdim, ),
        grid_spacing : tuple[float], (vdim, )
    Returns:
        weights: Tensor, in a shape of (N_1, N_2, ..., N_i, ...) for i from 1 to vdim
    '''
    w1ds = _create_w1ds(method, npoints, grid_spacing, device, dtype=dtype)
    return _create_w_block(w1ds)


def _create_u1d_GL(npt, device='cpu', dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Args:
        npt : integer
    Returns:
        a tensor of coefficients for interpolations at roots of npt-order GL polynomials
    '''
    roots, _ = roots_legendre(npt)
    roots_tensor = torch.as_tensor(roots, dtype=dtype, device=device)
    zeros_tensor = torch.zeros_like(roots_tensor)

    negative_mask = roots_tensor < 0

    neg_u = torch.stack(
        [-roots_tensor / 2, 1 + roots_tensor / 2, zeros_tensor], dim=-1)

    pos_u = torch.stack([
        zeros_tensor, 1 - roots_tensor / 2, roots_tensor / 2], dim=-1)

    u = torch.where(negative_mask.unsqueeze(-1), neg_u, pos_u)
    u.requires_grad = False
    return u


def _create_u1ds(method, npoints, device='cpu', dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Arguments:
        method : str
        npoints : (vdim, ) integers
    Return:
        (Tensor_1, Tensor_2, Tensor_3), a list in which each elment is
          a tensor of coefficients for linear interpolations.
    '''
    u1ds = []
    for ipt, npt in enumerate(npoints):
        u1ds.append(_create_u1d_GL(npt, device, dtype=dtype))
    return u1ds

def _create_u_block(u1ds, interpo_npts=3):
    '''
    To create a weight block for u in 3D
    Requirements:
        u1d in u1ds is Tensor with a shape of (npt, interpo_npts) where npt means n-point GL quad rule.
    Args:
        u1ds: list, (Tensor_1, Tensor_2, ..., Tensor_i, ...)
    Return:
        A tesnor in a shape of (N_1, N_2, ..., N_i, ..., interpo_npts, interpo_npts, ..., interpo_npts_i, ...)
    '''
    ndim = len(u1ds)
    for i in range(ndim):
        if u1ds[i].shape[-1] != interpo_npts:
            raise ValueError(
                f'u1ds must have last dimension of size '
                f'{interpo_npts} for linear interpolation')
    # [-1, 1, ..., 3, 1, ...]
    shape_new = [-1,] + (ndim-1) * [1,] + [interpo_npts,] + (ndim-1) * [1]
    ublock = u1ds[0].view(shape_new)
    for i in range(1, ndim, 1):
        shape_new = [1, ] * ndim + [1, ] * ndim
        shape_new[i] = -1
        shape_new[ndim+i] = interpo_npts
        ublock = ublock * u1ds[i].view(shape_new)

    return ublock


def create_u_block(method, npoints, device='cpu', dtype=DEFAULT_FLOAT_DTYPE):
    '''
    To create a weight block for u in 3D
    Args:
        method: string
        npoints: tuple[int], (vdim, )
        device
    Return:
        A tesnor in a shape of (N_1, N_2, ..., N_i, ..., 2, 2, ..., 2_i, ...)

    '''
    u1ds = _create_u1ds(method, npoints, device, dtype=dtype)
    return _create_u_block(u1ds)


def _create_wu_block(w, u):
    '''
    Args:
        w3d (L,M,N,...)
        u3d (L,M,N,...,2,2,2,...)
    Return:
        return w3d[i,j,k,...] * u3d{i,j,k,...,:,:,:,...]
    '''
    ndim = w.ndim
    shape = tuple(w.size()) + ndim * (1,)
    if w.ndim * 2 != u.ndim:
        raise ValueError('w and u does not match on their dimensions;'
                         f' w size: {w.size()}, u size: {u.size()}')
    return w.view(shape) * u


def create_wu_block(method, npoints, grid_spacing, device='cpu',
                    dtype=DEFAULT_FLOAT_DTYPE):
    w = create_w_block(method, npoints, grid_spacing, device, dtype=dtype)
    u = create_u_block(method, npoints, device, dtype=dtype)
    return _create_wu_block(w, u)


def create_grid1d(origin_1d, grid_spacing_1d, offset_1d, ngrid_1d : int,
                  device='cpu', dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Args:
        origin: float or 0-dimensional Tensor
        grid_spacing: float or 0-dimensional Tensor
        offset: Tensor of index_type, (N, )
        ngrid_1d: integer, number of grid points
    Returns:
        grid points

    FIXME: shp may be overflow?
    '''
    # if isinstance(shp_1d, Tensor):
    #     shp_1d = shp_1d.item()
    # if shp_1d > MAX_INDEX:
    #     raise ValueError(f'shp_1d={shp_1d} is larger than MAX_INDEX {MAX_INDEX}.')

    ngrid_idx_1d = torch.arange(ngrid_1d, dtype=index_dtype, requires_grad=False,
                              device=device) # (shp_1d, )
    idx_2d = offset_1d[:,None] + ngrid_idx_1d[None, :]
    grid_1d = compute_coordinate(idx_2d, (origin_1d,), (grid_spacing_1d,),
                                 device=device, dtype=dtype)
    return grid_1d

def _create_node1d_GL(npt, origin_1d, grid_spacing_1d, offset_1d, shp_1d,
                      device='cpu', dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Args:
        origin_1d : float or 0-dimensional Tensor
        grid_spacing_1d : float or 0-dimensional Tensor
        offset_1d : (Nsteps, ) Tensor of integers
        shp_1d : integer, the length of grid along the axis, in term of grid spacing
        npt : integer, npoints GL quad rule
    return:
        nodes (Nsteps, npt, shp_1d-1).

    FIXME: shp may be overflow?
        '''
    # if isinstance(shp_1d, Tensor):
    #     shp_1d = shp_1d.item()
    # if shp_1d > MAX_INDEX:
    #     raise ValueError(f'shp_1d={shp_1d} is larger than MAX_INDEX {MAX_INDEX}.')

    grid_1d = create_grid1d(origin_1d, grid_spacing_1d, offset_1d, shp_1d+1,
                            device, dtype=dtype)
    roots, _ = roots_legendre(npt)
    roots = to_tensor(roots, device=device, dtype=dtype)
    half_delta = (grid_1d[:, 1:] - grid_1d[:, :-1])/2. # (Nsteps, shp_1d-1)
    avg = (grid_1d[:,1:] + grid_1d[:,:-1])/2. # (Nsteps, shp_1d-1)
    node_1d = half_delta[:, None, :] * roots[None, :, None] \
        + avg[:, None, :] # (Nsteps, npt, shp_1d-1)
    return node_1d


def create_node1ds(method, npoints, origin, grid_spacing, offset, shape,
                   device='cpu', dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Args:
        method : str
        origin : tuple[float], in a size of (vdim, )
        grid_spaing : tuple[float], in a size of (vdim, )
        offset : Tensor, in a size of (Nbatch, vdim)
        shape : tuple[int], in a size of (vdim, ),
                the length of grid along each axis, in term of grid spacing, (shape1, shape2, ...)
        npoints : (vdim, ), n points for GL quad rule, (npoints1, npoints2, ...)
        device : device
    Return:
        a list in which each element is a tensor with a shape
        of (Nsteps, npoints1, npoints2, ..., shape1, shape2, ...)
    '''
    node1ds = []
    if method != 'gauss_legendre':
        raise NotImplementedError('Not implemented method but gauss legendre quadrature')
    for i in range(len(origin)):
        node1ds.append(
            _create_node1d_GL(npoints[i], origin[i], grid_spacing[i],
                              offset[:,i], shape[i], device, dtype=dtype)
        )
    return node1ds


def eval_qmodel(Q, X0, X1, Sigma, x, y, z, qmodel=qline_diff3D, **kwargs):
    '''
    Args:
        Q (Nsteps, )
        X0 (Nsteps, 3)
        X1 (Nsteps, 3)
        Sigma (Nsteps, 3)
        x, y, z are in a shape of (Nsteps, L/M/N, I/J/K-1)
    Return:
        q

    FIXME: to support axes with no nodes
    '''

    #FIXME: each axis must be associated with nodes of quadrature rules for now...
    charge = qmodel(Q, X0, X1, Sigma, x[:, :, None, None, :, None, None], # (Nsteps, L, 1, 1, I-1, 1, 1)
                    y[:, None, :, None, None, :, None], # (Nsteps, 1, M, 1, 1, J-1, 1)
                    z[:, None, None, :, None, None, :]) # (Nsteps, 1, 1, N, 1, 1, K-1)

    return charge


def eval_qeff(Q, X0, X1, Sigma, offset, shape, origin, grid_spacing,
              method, npoints, dtype=DEFAULT_FLOAT_DTYPE, **kwargs):
    '''
    Rasterize effective charge on a finite charge box.

    Validity assumptions:
        - The chosen raster box must be wide enough that truncation outside the
          box is small compared to the total integral. In practice this method
          is intended for boxes that extend to about +/- 3 sigma on each axis
          and retain at least about 99.5% of the total integral
          (equivalently, less than about 0.5% is lost outside the box).
        - The rasterization includes an interpolation step, so the summed
          rasterized charge does not exactly equal the exact integral over the
          finite box even when the box is fixed. Small tails outside the box
          are therefore required so that truncation error remains subdominant
          to the interpolation error.
        - The source model used for rasterization should be non-negative over
          the rasterized range.

    Args:
        Q (Nsteps, )
        X0 (Nsteps, 3)
        X1 (Nsteps, 3)
        Sigma (Nsteps, 3)
        offset (Nsteps, vdim)
        shape (vdim,)
        origin (vdim,)
        grid_spacing (vdim,)
        method: quadrature/interpolation backend name
        npoints (vdim, )
        dtype: floating compute dtype for node generation, model evaluation,
            and accumulation
        kwargs:
            qline_model: charge model used for non-point-like steps. The
                callable must take (Q, X0, X1, Sigma, x, y, z) and is passed
                through eval_qmodel() for broadcasting.
            qpoint_model: charge model used when too_short() classifies a step
                as point-like. The callable must have the same signature as
                qline_model.
            threshold: point-like classification threshold passed to too_short().
    Return:
        qeff, offset:
            qeff is a tensor of rasterized effective charge with shape
            (Nsteps, *shape), and offset is returned unchanged.

    FIXME:
       w block, u blocks supports vdim = any number,
       offset, shape, origin, they also support vdim = any number,
       Q, X0, X1, Sigma, their vdim is always 3.
    FIXME:
       vdim of npoints in principle can be different from offset, shape, origin, spacing.
       Now it is fixed to the same as others.
    '''
    if not isinstance(Q, torch.Tensor):
        raise ValueError('Q must be a torch.Tensor')
    device = Q.device
    Q = Q.to(device=device, dtype=dtype)
    X0 = X0.to(device=device, dtype=dtype)
    X1 = X1.to(device=device, dtype=dtype)
    Sigma = Sigma.to(device=device, dtype=dtype)
    offset = offset.to(device=device, dtype=index_dtype)
    shape = shape.to(device=device, dtype=index_dtype)
    grid_spacing = to_tensor(grid_spacing, device=device, dtype=dtype)
    origin = to_tensor(origin, device=device, dtype=dtype)

    qline_model = kwargs.get('qline_model', qline_diff3D)
    qpoint_model = kwargs.get('qpoint_model', qpoint_diff3D)
    threshold = kwargs.get('threshold', 0.1)

    # FIXME: dimensions are hard coded
    kernel = create_wu_block(method, npoints, grid_spacing, device, dtype=dtype)
    # it does not matter we flip at first or we multiply w and u at first
    kernel = torch.flip(kernel, [3, 4, 5])
    lmn = kernel.size()[:3]
    lmn_prod = lmn[0] * lmn[1] * lmn[2]
    rst = kernel.size()[3:]
    # out_channel, in_channel/groups, R, S, T
    kernel = kernel.view(lmn_prod, 1, rst[0], rst[1], rst[2])

    x, y, z = create_node1ds(method, npoints, origin, grid_spacing,
                             offset, shape, device, dtype=dtype)

    # FIXME: may update batch dimension in the future
    too_short_mask = too_short(X0, X1, Sigma, threshold=threshold)

    # Evaluate each charge model only on the steps that need it.
    #
    # This was previously a torch.where() over both models.  Both arms of
    # torch.where() are evaluated eagerly, so that computed the point model AND
    # the line model over the ENTIRE (nbatch, *lmn, *shape) grid for every step
    # and then threw one of them away per step -- roughly twice the peak memory
    # and twice the model-evaluation work needed.  It also allocated a
    # full-size zeros tensor that the torch.where() immediately replaced.
    #
    # Splitting by batch index costs one full-size buffer plus two partial
    # evaluations whose batch dimensions sum to the full batch.  index_put() is
    # used rather than in-place assignment so the result stays differentiable.
    # See tred-02x.4.
    point_idx = too_short_mask.nonzero(as_tuple=True)[0]
    line_idx = (~too_short_mask).nonzero(as_tuple=True)[0]

    def _subset(idx, qmodel):
        return eval_qmodel(Q[idx], X0[idx], X1[idx], Sigma[idx],
                           x[idx], y[idx], z[idx], qmodel=qmodel, **kwargs)

    if line_idx.numel() == 0:
        charge = _subset(point_idx, qpoint_model)
    elif point_idx.numel() == 0:
        charge = _subset(line_idx, qline_model)
    else:
        charge = torch.zeros((Q.size(0), *lmn, *shape),
                             dtype=dtype, device=device)
        charge = charge.index_put((point_idx,), _subset(point_idx, qpoint_model))
        charge = charge.index_put((line_idx,), _subset(line_idx, qline_model))

    charge = charge.view(Q.size(0), lmn_prod,  # batch, chanenel
                         shape[0], shape[1], shape[2])  # D1, D2, D3
    # padding='same' is symmetric only for an odd kernel.  The kernel's spatial
    # extent here is the interpolation stencil (kernel.size()[3:], i.e. rst),
    # NOT the quadrature order npoints (which is kernel.size()[:3], lmn).  The
    # stencil is 3 for every supported configuration: _create_u1d_GL() returns
    # a (npt, 3) tensor whatever npt is, and _create_u_block() raises if handed
    # anything else.  So the odd-kernel requirement holds by construction.
    # Asserted in tests/effq/test_padding_stencil.py; if the stencil ever
    # becomes even, that padding must change to an explicit asymmetric pad.
    charge = torch.nn.functional.conv3d(charge, kernel, padding='same',
                                        groups=lmn_prod)
    charge = charge.sum(dim=1)
    return charge, offset


# Fraction of a grid cell below which a Gaussian cannot be resolved by the
# default 2-point Gauss-Legendre rule.  See apply_min_sigma().
MIN_SIGMA_CELL_FRACTION = 0.5

_warned_min_sigma = False


def apply_min_sigma(Sigma, grid_spacing, min_sigma='auto',
                    dtype=DEFAULT_FLOAT_DTYPE):
    '''
    Raise Sigma to a per-axis floor so the charge model stays resolvable.

    eval_qeff() integrates the charge model with an npoints-per-cell
    Gauss-Legendre rule.  When sigma is small compared to the cell the nodes
    can miss the peak, and the rasterized total is wrong -- silently, and in
    either direction.  Measured for the default npoints=(2,2,2), sweeping a
    point deposit across a cell and taking the worst case:

        sigma/spacing   worst error
             1.00          0.8%     <- this is the +/-3 sigma box truncation
             0.75          0.6%
             0.50          0.8%
             0.40          3.9%
             0.30         13.7%
             0.25         26.1%

    So half a cell is where the quadrature error disappears into the box
    truncation error already present.  Below about an eighth of a cell the
    result collapses to zero, and at sigma=0 it is NaN.

    Arguments:
        min_sigma: 'auto' for MIN_SIGMA_CELL_FRACTION * grid_spacing per axis,
                   which is the default; a scalar or per-axis sequence to set
                   the floor explicitly; or None/0 to disable it and accept
                   the unresolved-quadrature error.

    Raising the order instead would work -- n=6 holds down to about an eighth
    of a cell -- but the quadrature block scales as prod(npoints), so n=6 costs
    27x the memory and compute of n=2 in 3D, and it is needed precisely for the
    narrow depos that are cheapest to get right by widening.

    A floor is an approximation: it broadens charge that is genuinely narrower
    than the grid can represent.  That is preferable to losing it. In
    production the grid is the pitch divided by the impact-position count, and
    post-drift sigma is normally several cells, so this should rarely engage.

    See tred-xsj.
    '''
    global _warned_min_sigma

    if min_sigma is None:
        return Sigma

    if isinstance(min_sigma, str):
        if min_sigma != 'auto':
            raise ValueError(f"min_sigma must be 'auto', a number, a sequence "
                             f"or None, got {min_sigma!r}")
        floor = to_tensor(grid_spacing, device=Sigma.device, dtype=dtype)
        floor = floor.abs() * MIN_SIGMA_CELL_FRACTION
    else:
        floor = to_tensor(min_sigma, device=Sigma.device, dtype=dtype)

    if floor.ndim == 0:
        floor = floor.expand(Sigma.shape[-1])
    if not bool(torch.any(floor > 0)):
        return Sigma

    floored = torch.maximum(Sigma, floor.unsqueeze(0))

    if not _warned_min_sigma and bool(torch.any(floored > Sigma)):
        _warned_min_sigma = True
        logger.warning(
            'raster: sigma raised to the resolvable floor on at least one '
            'axis (floor %s). Below about half a grid cell the quadrature '
            'silently loses charge; pass min_sigma=None to disable this. '
            'Warned once per process.', floor.tolist())

    return floored


def compute_qeff(Q, X0, X1, Sigma, n_sigma, origin, grid_spacing, method, npoints, **kwargs):
    '''
    Compute a charge box from the step geometry and then rasterize the
    effective charge within that box.

    Validity assumptions:
        - The charge box selected through n_sigma should be large enough that
          the omitted tail outside the box is small. In practice this means
          choosing a box close to +/- 3 sigma on each axis and retaining at
          least about 99.5% of the total integral
          (equivalently, less than about 0.5% lies outside the box).
        - The rasterization includes an interpolation step, so the summed
          rasterized charge only approximates the exact integral over the
          chosen finite box. The tail outside the box should therefore be
          sufficiently small that truncation error does not dominate this
          interpolation error.
        - The source model should remain non-negative over the rasterized
          range.

    Args:
        Q (Nsteps, )
        X0 (Nsteps, 3)
        X1 (Nsteps, 3)
        Sigma (Nsteps, 3)
        n_sigma: charge-box half-width in units of sigma for each axis
        origin (vdim,)
        grid_spacing (vdim,)
        method: quadrature/interpolation backend name
        npoints (vdim,)
        kwargs: forwarded to compute_charge_box() and eval_qeff(). If either
            recenter=True or skippad=True is requested, both are enabled to
            keep the charge box and raster output aligned.

    Return:
        qeff, offset:
            qeff is the rasterized effective charge tensor and offset is the
            lower grid index of the computed charge box.
    '''
    if kwargs.get('recenter', False) or kwargs.get('skippad', False):
        if not kwargs.get('recenter', False):
            kwargs['recenter'] = True
            logger.warning('Option "recenter" is overwritten to True, complying with option "skippad"')
        if not kwargs.get('skippad', False):
            kwargs['skippad'] = True
            logger.warning('Option "skippad" is overwritten to True, complying with option "recenter"')
    dtype = kwargs.pop('dtype', DEFAULT_FLOAT_DTYPE)
    Q = Q.to(dtype=dtype)
    X0 = X0.to(dtype=dtype)
    X1 = X1.to(dtype=dtype)
    Sigma = Sigma.to(dtype=dtype)
    Sigma = apply_min_sigma(Sigma, grid_spacing,
                            kwargs.pop('min_sigma', 'auto'), dtype=dtype)
    offset, shape = compute_charge_box(X0, X1, Sigma, n_sigma, origin, grid_spacing,
                                       dtype=dtype, **kwargs)
    qeff, offset = eval_qeff(Q, X0, X1, Sigma, offset, shape, origin, grid_spacing,
                             method, npoints, dtype=dtype, **kwargs)
    return qeff, offset
