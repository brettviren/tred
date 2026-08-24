'''
Gradient flow through chunking.

chunkify() fills its envelope with an in-place indexed assignment;
chunkify_bins() does the same job with Tensor.index_put().

The end_to_end branch added the latter on the premise that the former "breaks
the autograd graph".  Measured here, that is not so: in the normal path both
propagate gradients and produce identical grads.  The difference shows up only
when the envelope's data is itself a grad-requiring leaf, where the in-place
write raises.  Tests below cover both facts, so neither claim can drift.

Reworked from the branch, which had a single hardcoded 2x2 case plus leftover
prints and commented-out assertions.  See tred-4pz.7 and tred-4pz.11.
'''

import pytest
import torch

from tred.blocking import Block
from tred.chunking import accumulate
from tred.sparse import chunkify, chunkify_bins

F32 = torch.float32


def _block(requires_grad=False):
    location = torch.tensor([[1, 1], [1, 2]])
    data = torch.tensor([[[2., 3.], [1., 0.]],
                         [[4., 0.], [0., 1.]]],
                        dtype=F32, requires_grad=requires_grad)
    return Block(location=location, data=data), data


# --------------------------------------------------------------------------
# equivalence with the in-place original
# --------------------------------------------------------------------------

@pytest.mark.parametrize('shape', [(2, 2), (4, 4), (3, 3), (1, 1), (2, 4)])
def test_chunkify_bins_matches_chunkify(shape):
    block, _ = _block()
    plain = accumulate(chunkify(block, shape))
    bins = accumulate(chunkify_bins(block, shape))
    assert torch.equal(plain.location, bins.location)
    assert torch.allclose(plain.data, bins.data, atol=1e-6, rtol=1e-6)


def test_chunkify_bins_matches_chunkify_on_a_bigger_random_block():
    torch.manual_seed(20260824)
    location = torch.tensor([[0, 0, 0], [2, 1, 3], [5, 4, 1], [2, 1, 3]])
    data = torch.rand((4, 3, 3, 3), dtype=F32)
    block = Block(location=location, data=data)
    plain = accumulate(chunkify(block, (3, 3, 3)))
    bins = accumulate(chunkify_bins(block, (3, 3, 3)))
    assert torch.equal(plain.location, bins.location)
    assert torch.allclose(plain.data, bins.data, atol=1e-6, rtol=1e-6)


def test_chunkify_bins_conserves_content():
    block, _ = _block()
    out = accumulate(chunkify_bins(block, (4, 4)))
    assert out.data.sum().item() == pytest.approx(block.data.sum().item())


# --------------------------------------------------------------------------
# the reason it exists
# --------------------------------------------------------------------------

def test_both_variants_propagate_gradients_identically():
    '''
    Measured, against the expectation this file was written under: in the
    normal path chunkify() ALREADY propagates gradients correctly.  Its
    in-place envelope write is recorded by autograd and the two give the same
    grad.  chunkify_bins() is about robustness, not capability.
    '''
    grads = []
    for fn in (chunkify, chunkify_bins):
        block, data = _block(requires_grad=True)
        out = accumulate(fn(block, (4, 4)))
        assert out.data.requires_grad
        out.data.square().sum().backward()
        grads.append(data.grad.clone())
    assert torch.equal(grads[0], grads[1])


def test_index_put_survives_a_grad_requiring_envelope():
    '''
    The case that does differ.  fill_envelope() writes through a flatten()
    view, so a grad-requiring leaf envelope raises "a view of a leaf Variable
    that requires grad is being used in an in-place operation".
    '''
    from tred.sparse import SGrid, fill_envelope, fill_envelope_index_put

    def attempt(fill):
        block, data = _block(requires_grad=True)
        envelope = SGrid(torch.tensor([4, 4])).envelope(block)
        envelope.set_data(torch.zeros(envelope.size(), requires_grad=True))
        return fill(envelope, block)

    with pytest.raises(RuntimeError):
        attempt(fill_envelope)
    out = attempt(fill_envelope_index_put)
    out.data.sum().backward()


def test_gradient_matches_a_dense_reference():
    '''
    d/dx of sum(chunked^2) must equal 2*chunked at the positions the block
    occupies.  Built against an explicit dense placement rather than a second
    call to the same machinery.
    '''
    block, data = _block(requires_grad=True)
    out = accumulate(chunkify_bins(block, (4, 4)))
    out.data.square().sum().backward()

    # dense equivalent: place each sub-block into a 4x4 grid and sum overlaps
    dense = torch.zeros((4, 4), dtype=F32, requires_grad=False)
    ref = data.detach().clone().requires_grad_(True)
    acc = torch.zeros((4, 4), dtype=F32)
    for i, (r, c) in enumerate(block.location.tolist()):
        pad = torch.zeros((4, 4), dtype=F32)
        pad[r:r + 2, c:c + 2] = ref[i]
        acc = acc + pad
    acc.square().sum().backward()

    assert data.grad is not None
    assert torch.allclose(data.grad, ref.grad, atol=1e-5), \
        f'{data.grad} != {ref.grad}'


def test_gradient_is_finite_and_nonzero():
    block, data = _block(requires_grad=True)
    accumulate(chunkify_bins(block, (4, 4))).data.sum().backward()
    assert torch.isfinite(data.grad).all()
    assert data.grad.abs().sum() > 0


def test_chunksum_differential_method_flows_gradients():
    from tred.graph import ChunkSum
    block, data = _block(requires_grad=True)
    out = ChunkSum((4, 4), method='chunksum_differential')(block)
    assert out.data.requires_grad
    out.data.sum().backward()
    assert torch.isfinite(data.grad).all()


def test_chunksum_differential_agrees_with_chunksum():
    from tred.graph import ChunkSum
    block, _ = _block()
    plain = ChunkSum((4, 4), method='chunksum')(block)
    diff = ChunkSum((4, 4), method='chunksum_differential')(block)
    assert torch.equal(plain.location, diff.location)
    assert torch.allclose(plain.data, diff.data, atol=1e-6, rtol=1e-6)


def test_unknown_chunksum_method_lists_the_valid_ones():
    from tred.graph import ChunkSum
    with pytest.raises(ValueError, match='chunksum_differential'):
        ChunkSum((4, 4), method='nope')
