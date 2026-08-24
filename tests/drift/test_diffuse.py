import torch
import pytest

from tred.drift import diffuse

# Assume the diffuse function is already defined

def test_diffuse_scalar_diffusion_no_sigma():
    # diffusion provided as a scalar (not a tensor)
    dt = torch.tensor([1.0, 2.0, 3.0])
    diffusion = 5.0
    # Expected: sigma = sqrt(2 * diffusion * dt)
    expected = torch.sqrt(2 * 5.0 * dt)
    result = diffuse(dt, diffusion)
    # Because diffusion was scalar, the function sets squeeze=True so that
    # the extra dimension is removed.
    assert torch.allclose(result, expected)

def test_diffuse_vector_diffusion_no_sigma():
    # diffusion provided as a 1D tensor (vector)
    dt = torch.tensor([1.0, 2.0])
    diffusion = torch.tensor([4.0, 5.0, 6.0])
    # Expected: sigma = sqrt(2 * dt[:, None] * diffusion[None, :])
    expected = torch.sqrt(2 * dt[:, None] * diffusion[None, :])
    result = diffuse(dt, diffusion)
    # In this case diffusion is already a tensor, so squeeze remains False and
    # the output shape is (npts, diffusion_length)
    assert torch.allclose(result, expected)

def test_diffuse_with_sigma():
    # Test the case where an initial sigma is provided.
    dt = torch.tensor([1.0, 2.0])
    diffusion = 1.0  # scalar diffusion
    sigma_initial = torch.tensor([0.5, 1.0])  # one sigma per time point (1D)
    # Expected: sigma = sqrt(2*diffusion*dt + sigma_initial^2)
    expected = torch.sqrt(2 * 1.0 * dt + sigma_initial**2)
    result = diffuse(dt, diffusion, sigma_initial)
    # For scalar diffusion, squeeze=True so that the result is squeezed to shape (npts,)
    assert result.shape == sigma_initial.shape
    assert torch.allclose(result, expected)

def test_diffuse_negative_dt():
    # Test with a negative dt causing the argument of sqrt to be negative.
    dt = torch.tensor([-1.0])
    diffusion = 1.0
    result = diffuse(dt, diffusion)
    # sqrt(2*1*(-1)) = sqrt(-2) returns nan, which the function then sets to 0.
    expected = torch.tensor(0.0)
    assert torch.allclose(result, expected)

def test_dt_shape_exception():
    # dt must be 1D. Here we pass a 2D tensor.
    dt = torch.tensor([[1.0, 2.0]])
    diffusion = 1.0
    with pytest.raises(ValueError):
        diffuse(dt, diffusion)

def test_diffusion_shape_exception():
    # diffusion must be 1D. Here we pass a 2D tensor.
    dt = torch.tensor([1.0])
    diffusion = torch.tensor([[1.0]])
    with pytest.raises(ValueError):
        diffuse(dt, diffusion)

def test_sigma_shape_exception():
    # sigma's shape should span the same number of dimensions as diffusion (which is 1 for 1D diffusion)
    # Here we provide a sigma with an extra dimension (2D instead of 1D)
    dt = torch.tensor([1.0, 2.0])
    diffusion = 1.0  # so diffusion is internally converted to a 1D tensor of length 1
    sigma_wrong = torch.tensor([[0.5, 1.0]])  # shape (1, 2) does not match the expected 1D shape for sigma
    with pytest.raises(ValueError):
        diffuse(dt, diffusion, sigma_wrong)


# Negative dt means the drift "backed up".  diffuse() treats such entries as
# dt=0, consistent with absorb() leaving charge unchanged for negative dt.
#
# This replaced an earlier scheme where negative dt shrank the initial sigma
# and, once 2*D*dt + sigma^2 went negative, the resulting NaN was masked to
# zero.  See tred-4pz.6.

def test_negative_dt_preserves_initial_sigma():
    dt = torch.tensor([1.0, -0.5, -5.0])
    sigma0 = torch.tensor([0.3, 0.3, 0.3])
    result = diffuse(dt, 5.0, sigma0)
    # the positive entry diffuses normally
    assert torch.isclose(result[0], torch.sqrt(torch.tensor(2 * 5.0 * 1.0 + 0.09)))
    # the negative entries keep their initial width rather than collapsing
    assert torch.allclose(result[1:], sigma0[1:])


def test_negative_dt_without_initial_sigma_is_zero():
    # With no initial width there is nothing to preserve, so a backed-up drift
    # still yields zero.  Masking the NaN and clamping dt agree here, which is
    # why this case alone does not distinguish the two schemes.
    dt = torch.tensor([1.0, -0.5])
    result = diffuse(dt, 5.0)
    assert torch.isclose(result[0], torch.sqrt(torch.tensor(2 * 5.0 * 1.0)))
    assert result[1] == 0.0


def test_negative_dt_never_produces_nan():
    # A zero or NaN sigma is dangerous downstream: compute_qeff() returns
    # all-zero charge for a very narrow sigma and NaN for exactly zero.
    dt = torch.tensor([-1e-9, -1.0, -1e9])
    assert not torch.isnan(diffuse(dt, 5.0)).any()
    assert not torch.isnan(diffuse(dt, 5.0, torch.tensor([0.3, 0.3, 0.3]))).any()


def test_diffusion_never_narrows_the_initial_sigma():
    # The invariant that actually separates the two schemes: diffusion can only
    # ADD width.  The old scheme narrowed it -- smoothly at first, then to zero
    # once 2*D*dt + sigma^2 went negative, which for D=5 and sigma=0.3 happens
    # below dt = -0.009.  Testing either side of dt=0 does NOT distinguish the
    # schemes, because the old one is still smooth there.
    D, sigma0 = 5.0, 0.3
    dt = torch.tensor([-10.0, -1.0, -0.01, -0.009, -0.001, 0.0, 1.0],
                      dtype=torch.float64)
    sig = torch.full_like(dt, sigma0)
    result = diffuse(dt, D, sig)
    assert torch.all(result >= sigma0 - 1e-12), \
        f'diffusion narrowed the initial sigma: {result.tolist()}'
    # and it is monotone non-decreasing in dt
    assert torch.all(result[1:] - result[:-1] >= -1e-12)


def test_gradient_flows_through_negative_dt():
    # The in-place NaN mask this replaced broke autograd, which is what blocked
    # the differentiable path in the end_to_end branch.
    dt = torch.tensor([1.0, -0.5], dtype=torch.float64, requires_grad=True)
    out = diffuse(dt, 5.0, torch.tensor([0.3, 0.3], dtype=torch.float64))
    out.sum().backward()
    assert not torch.isnan(dt.grad).any()
    assert dt.grad[0] > 0.0      # positive dt contributes
    assert dt.grad[1] == 0.0     # clamped entry contributes nothing
