'''
End-to-end differentiability of the simulation graph.

The end_to_end branch flipped four global defaults on the premise that they
blocked gradient flow: param() requires_grad, a trainable Drifter.lifetime,
chunking.accumulate() pruning, and the integer cast in drift.absorb().

Measured, none of them does -- see tred-4pz.12.  A lifetime gradient reaches
the loss through the default configuration.  These tests pin that, so the
four defaults cannot be changed on the old premise, and so a real regression in
differentiability is caught.

The one change that WAS required is raster.steps.to_tensor() no longer forcing
requires_grad=False (tred-4pz.9); without it these tests fail.
'''

import pytest
import torch
import torch.nn as nn

from tred.chunking import accumulate
from tred.blocking import Block
from tred.drift import absorb
from tred.graph import (Charge, ChunkSum, Current, Drifter, LacedConvo, Raster,
                        Sim, param)

GRID = [0.4, 0.4, 0.05]
VELOCITY = -1.6
NIMPER = 2


def _steps():
    time = torch.tensor([0.0, 0.0, 1.0])
    charge = torch.tensor([1000.0, 800.0, 1200.0])
    tail = torch.tensor([[2.0, 1.0, 1.0], [3.0, 0.5, 0.5], [1.0, 2.0, 2.0]])
    head = torch.tensor([[2.4, 1.2, 1.1], [3.0, 0.5, 0.5], [1.8, 2.6, 2.3]])
    return time, charge, tail, head


def _response():
    ax = torch.linspace(0, 1, 3 * NIMPER)
    at = torch.linspace(0, 1, 24)
    return (torch.exp(-3 * ax)[:, None, None]
            * torch.exp(-3 * ax)[None, :, None]
            * torch.exp(-2 * at)[None, None, :])


def _sim(lifetime, chunk_method='chunksum'):
    '''Build a Sim whose lifetime is trainable, the way graph_opt does it.'''
    lacing = torch.tensor([NIMPER, NIMPER, 1])
    drifter = Drifter(diffusion=[0.1, 0.05, 0.05], lifetime=10.0,
                      velocity=VELOCITY, target=0.0)
    drifter.lifetime = nn.Parameter(torch.tensor(float(lifetime)),
                                    requires_grad=True)
    sim = Sim(
        Charge(drifter, Raster(velocity=VELOCITY, grid_spacing=GRID),
               ChunkSum((4 * NIMPER, 4 * NIMPER, 16), method=chunk_method)),
        Current(LacedConvo(lacing, o_shape=(8, 8, 40)),
                ChunkSum((4, 4, 8), method='chunksum_inplace_v2')))
    return sim, drifter


def test_lifetime_gradient_reaches_the_loss_through_the_full_graph():
    '''The claim the whole end_to_end epic rests on, on default settings.'''
    response, (time, charge, tail, head) = _response(), _steps()
    with torch.no_grad():
        truth = _sim(10.0)[0](response, time, charge, tail, head).data.clone()

    sim, drifter = _sim(12.0)
    out = sim(response, time, charge, tail, head)
    loss = nn.MSELoss()(out.data, truth)
    loss.backward()

    assert drifter.lifetime.grad is not None
    assert torch.isfinite(drifter.lifetime.grad)
    assert drifter.lifetime.grad != 0.0


@pytest.mark.parametrize('method', ['chunksum', 'chunksum_differential'])
def test_both_chunksum_methods_give_the_same_gradient(method):
    response, (time, charge, tail, head) = _response(), _steps()
    with torch.no_grad():
        truth = _sim(10.0)[0](response, time, charge, tail, head).data.clone()

    sim, drifter = _sim(12.0, method)
    nn.MSELoss()(sim(response, time, charge, tail, head).data, truth).backward()
    # value recorded from the default path; both methods must agree with it
    assert float(drifter.lifetime.grad) == pytest.approx(0.048494, rel=1e-3)


def test_a_step_of_gradient_descent_moves_lifetime_toward_the_truth():
    '''Weak but end-to-end: the sign of the gradient has to be useful.'''
    response, (time, charge, tail, head) = _response(), _steps()
    with torch.no_grad():
        truth = _sim(10.0)[0](response, time, charge, tail, head).data.clone()

    sim, drifter = _sim(12.0)
    before = drifter.lifetime.detach().clone()
    opt = torch.optim.SGD([drifter.lifetime], lr=1.0)
    nn.MSELoss()(sim(response, time, charge, tail, head).data, truth).backward()
    opt.step()
    assert drifter.lifetime.detach() < before, 'step moved away from the truth'


# --------------------------------------------------------------------------
# the four defaults the branch changed, each shown not to block gradients
# --------------------------------------------------------------------------

def test_param_defaults_to_no_grad_and_that_is_fine():
    '''
    The branch made param() produce requires_grad=True parameters. Nothing
    needs that: graph_opt installs its own trainable tensor, as _sim does.
    '''
    p = param([1.0, 2.0])
    assert p.requires_grad is False


def test_constant_lifetime_can_still_be_made_trainable():
    drifter = Drifter(diffusion=[0.1, 0.05, 0.05], lifetime=10.0,
                      velocity=VELOCITY, target=0.0)
    assert not isinstance(drifter.lifetime, nn.Parameter)
    drifter.lifetime = nn.Parameter(torch.tensor(12.0), requires_grad=True)
    time, charge, tail, _ = _steps()
    drifter(time, charge, tail)[2].sum().backward()
    assert drifter.lifetime.grad is not None


def test_accumulate_pruning_does_not_block_gradients():
    '''
    The branch deleted the zero-pruning in accumulate() for differentiability.
    Boolean-mask indexing is differentiable; the pruning can stay.
    '''
    loc = torch.tensor([[0, 0], [0, 0], [2, 2]])
    base = torch.zeros((3, 2, 2), requires_grad=True)
    data = base + torch.tensor([[[0., 0.], [0., 0.]],
                                [[0., 0.], [0., 0.]],
                                [[1., 2.], [3., 4.]]])
    out = accumulate(Block(location=loc, data=data))
    assert out.nbatches == 1                      # pruning did happen
    out.data.sum().backward()
    assert torch.isfinite(base.grad).all()


def test_absorb_int_cast_blocks_charge_but_not_lifetime():
    '''
    The cast detaches charge, so d/d(charge) is unavailable -- but lifetime is
    what gets fitted, and that gradient survives. Documents the real boundary
    so tred-4pz.16 is not solved speculatively.
    '''
    lifetime = torch.tensor(10.0, requires_grad=True)
    absorb(torch.tensor([1000.0]), torch.tensor([5.0]), lifetime).sum().backward()
    assert lifetime.grad is not None and torch.isfinite(lifetime.grad)

    q = torch.tensor([1000.0], requires_grad=True)
    out = absorb(q, torch.tensor([5.0]), torch.tensor(10.0))
    assert not out.requires_grad, 'the int cast should still detach charge'
