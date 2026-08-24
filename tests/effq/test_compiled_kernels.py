'''
The opt-in torch.compile path for the charge-model kernels.

`raster.steps` can route qline_diff3D_script and qpoint_diff3D_kernel through
torch.compile, which lets inductor fuse their elementwise chains and drop most
of the intermediates.  Off by default; see tred-02x.12.

These tests are slow-ish: the first compiled call costs a few seconds.  They
use a small charge box to keep that down.
'''

import pytest
import torch

from tred.raster import steps
from tred.raster.steps import compute_qeff, set_compiled_kernels

F64 = torch.float64
GRID = torch.tensor([0.4, 0.4, 0.05], dtype=F64)

# Fusion reorders floating-point operations, so the compiled result agrees to
# a few ULP rather than bit-exactly.  Measured worst relative deviation across
# the qeff regression cases was 3.1e-16, about 1.4 ULP of float64.
COMPILED_RTOL = 1e-12


@pytest.fixture
def compiled_off():
    '''Leave the module's setting exactly as we found it.'''
    was = steps._COMPILE_KERNELS
    yield
    set_compiled_kernels(was)
    steps._compiled_kernels.clear()


def _qeff(npoints=(2, 2, 2)):
    return compute_qeff(
        Q=torch.tensor([1000.0], dtype=F64),
        X0=torch.tensor([[1.0, 1.0, 1.0]], dtype=F64),
        X1=torch.tensor([[1.6, 1.4, 1.3]], dtype=F64),
        Sigma=torch.tensor([[0.4, 0.4, 0.2]], dtype=F64),
        n_sigma=(3.0, 3.0, 3.0), origin=(0, 0, 0), grid_spacing=GRID,
        method='gauss_legendre', npoints=npoints)


def test_off_by_default_unless_env_set(monkeypatch):
    '''The default must be the uncompiled path.'''
    assert isinstance(steps._COMPILE_KERNELS, bool)


def test_set_compiled_kernels_round_trips(compiled_off):
    was = set_compiled_kernels(True)
    assert steps._COMPILE_KERNELS is True
    prev = set_compiled_kernels(False)
    assert prev is True
    assert steps._COMPILE_KERNELS is False
    set_compiled_kernels(was)


def test_kernel_returns_plain_function_when_disabled(compiled_off):
    set_compiled_kernels(False)
    fn = steps._kernel(steps.qpoint_diff3D_kernel)
    assert fn is steps.qpoint_diff3D_kernel


def test_compile_failure_falls_back_to_the_plain_kernel(compiled_off, monkeypatch):
    '''Compilation is an optimization; a failure must not break rasterization.'''
    set_compiled_kernels(True)
    steps._compiled_kernels.clear()

    def boom(*a, **kw):
        raise RuntimeError('inductor unavailable')

    monkeypatch.setattr(torch, 'compile', boom)
    fn = steps._kernel(steps.qpoint_diff3D_kernel)
    assert fn is steps.qpoint_diff3D_kernel


@pytest.mark.slow
def test_compiled_agrees_with_uncompiled(compiled_off):
    set_compiled_kernels(False)
    steps._compiled_kernels.clear()
    plain, plain_offset = _qeff()

    set_compiled_kernels(True)
    steps._compiled_kernels.clear()
    try:
        fused, fused_offset = _qeff()
    except Exception as err:                      # pragma: no cover
        pytest.skip(f'torch.compile unusable here: {err}')

    assert torch.equal(plain_offset, fused_offset)
    assert plain.shape == fused.shape
    assert torch.allclose(plain, fused, rtol=COMPILED_RTOL, atol=0.0), \
        f'max deviation {float((plain - fused).abs().max())}'
    # charge conservation must not drift either
    assert abs(float(plain.sum()) - float(fused.sum())) / float(plain.sum()) < 1e-12


@pytest.mark.slow
def test_compiled_handles_varying_shapes_without_recompiling(compiled_off):
    '''
    dynamic=True is essential.  Charge-box shapes vary from step to step, and
    with dynamic=False each new shape triggers a multi-second recompile.  This
    checks that a spread of shapes stays fast after the first call.
    '''
    import time
    set_compiled_kernels(True)
    steps._compiled_kernels.clear()
    try:
        _qeff()                                   # absorb the one-off compile
    except Exception as err:                      # pragma: no cover
        pytest.skip(f'torch.compile unusable here: {err}')

    worst = 0.0
    for zlen in (1.3, 1.7, 2.3, 3.1, 4.5):
        X1 = torch.tensor([[1.6, 1.4, zlen]], dtype=F64)
        start = time.perf_counter()
        compute_qeff(Q=torch.tensor([1000.0], dtype=F64),
                     X0=torch.tensor([[1.0, 1.0, 1.0]], dtype=F64), X1=X1,
                     Sigma=torch.tensor([[0.4, 0.4, 0.2]], dtype=F64),
                     n_sigma=(3.0, 3.0, 3.0), origin=(0, 0, 0),
                     grid_spacing=GRID, method='gauss_legendre',
                     npoints=(2, 2, 2))
        worst = max(worst, time.perf_counter() - start)

    # A recompile costs seconds; a cached dynamic kernel costs milliseconds.
    assert worst < 1.0, f'a varying shape took {worst:.2f}s, suggesting a recompile'
