'''
Compare the current code against recorded numerical baselines.

A failure here is not automatically a bug.  It means behaviour changed, and the
change now has to be either fixed or accepted.  To accept it, re-record with

    python tests/regression/baseline_cases.py --write <case_prefix>

and commit the fixture diff alongside the code change, so the review shows
exactly which numbers moved.

See tred-4pz.5.
'''

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import baseline_cases as bc  # noqa: E402

# Tolerances.  The chain is float64 end to end apart from the graph nodes,
# which default to float32, so a strict equality check would be brittle across
# BLAS versions.  These are tight enough that a real behaviour change shows up.
RTOL = 1e-10
ATOL = 1e-12

# Peak-memory figures are indicative, not reproducible to the byte.  They are
# recorded for tred-02x to measure against but are not asserted here.
UNASSERTED_PREFIX = '_'


def _baseline_path(name):
    return os.path.join(bc.BASELINE_DIR, f'{name}.pt')


def _load(name):
    path = _baseline_path(name)
    if not os.path.exists(path):
        pytest.skip(f'no baseline recorded for {name}; run '
                    f'"python tests/regression/baseline_cases.py --write {name}"')
    return torch.load(path, weights_only=True)


@pytest.mark.parametrize('name', sorted(bc.CASES))
def test_matches_baseline(name):
    expected = _load(name)
    actual = bc.run_case(name)

    unexpected = set(actual) - set(expected)
    missing = set(expected) - set(actual)
    assert not unexpected, f'{name}: keys not in baseline: {sorted(unexpected)}'
    assert not missing, f'{name}: keys missing from result: {sorted(missing)}'

    for key in sorted(expected):
        if key.startswith(UNASSERTED_PREFIX):
            continue
        exp = torch.as_tensor(expected[key])
        got = torch.as_tensor(actual[key])

        assert got.shape == exp.shape, \
            f'{name}.{key}: shape {tuple(got.shape)} != baseline {tuple(exp.shape)}'

        # NaN is meaningful in some baselines (see tred-xsj), so compare NaN
        # placement explicitly rather than letting allclose swallow it.
        exp_nan, got_nan = torch.isnan(exp), torch.isnan(got)
        assert torch.equal(exp_nan, got_nan), \
            f'{name}.{key}: NaN pattern changed'

        if exp.is_floating_point():
            assert torch.allclose(got[~got_nan], exp[~exp_nan],
                                  rtol=RTOL, atol=ATOL), (
                f'{name}.{key}: values changed\n'
                f'  baseline: {exp[~exp_nan].flatten()[:8]}\n'
                f'  current : {got[~got_nan].flatten()[:8]}')
        else:
            assert torch.equal(got, exp), \
                f'{name}.{key}: integer values changed\n' \
                f'  baseline: {exp.flatten()[:8]}\n  current : {got.flatten()[:8]}'


def test_every_case_has_a_baseline():
    '''A case added without a recorded fixture would otherwise silently skip.'''
    missing = [n for n in sorted(bc.CASES) if not os.path.exists(_baseline_path(n))]
    assert not missing, (
        f'cases without baselines: {missing}\n'
        f'run: python tests/regression/baseline_cases.py --write')


def test_metadata_present():
    path = os.path.join(bc.BASELINE_DIR, 'metadata.json')
    assert os.path.exists(path), 'baselines/metadata.json missing'
    with open(path) as fp:
        meta = json.load(fp)
    assert meta['seed'] == bc.SEED
    assert meta['grid_spacing'] == bc.GRID
    assert sorted(meta['cases']) == sorted(bc.CASES), \
        'metadata case list is stale; re-record the baselines'


# --------------------------------------------------------------------------
# Properties that should hold regardless of the recorded numbers.  These
# encode intent, where the fixtures above only encode current behaviour.
# --------------------------------------------------------------------------

def test_accumulate_prunes_all_zero_content():
    '''
    Pins the pruning behaviour the end_to_end branch removes.

    When tred-4pz.15 makes pruning conditional this must still hold at the
    default setting; only an explicit prune=False may return the zero chunks.
    '''
    result = bc.run_case('accumulate_all_zero')
    assert result['acc_nbatches'].item() == 0, \
        'accumulate() no longer prunes all-zero chunks at its default setting'


def test_chunksum_methods_conserve_total():
    '''
    The four ChunkSum methods legitimately differ in chunk count but must all
    conserve the summed content.
    '''
    result = bc.run_case('chunksum_methods')
    totals = {k: v.item() for k, v in result.items() if k.endswith('_sum')}
    reference = totals['chunksum_sum']
    for name, total in totals.items():
        assert abs(total - reference) < 1e-6, \
            f'{name}={total} disagrees with chunksum={reference}'


def test_charge_is_conserved_at_every_width():
    '''
    Charge conservation now holds across the whole sigma range.

    It used to hold only when sigma was comparable to the grid pitch; below
    about half a cell the quadrature silently lost charge, down to returning
    exactly zero and then NaN (tred-xsj).  The resolvable-sigma floor fixed
    that, so this asserts the general property rather than a carve-out.
    '''
    for name, true_q in (('qeff_wide_diffusion', 1000.0),
                         ('qeff_point_like', 1000.0),
                         ('qeff_anisotropic', 1000.0),
                         ('qeff_narrow_sigma_charge_loss', 1000.0),
                         ('qeff_mixed_batch', 1500.0)):
        got = bc.run_case(name)['sum'].item()
        assert abs(got - true_q) / true_q < 0.03, \
            f'{name}: rasterized {got} of {true_q}'


def test_narrow_sigma_still_fails_without_the_floor():
    '''The floor must be doing the work, not coincidence.'''
    raw = bc.run_case('qeff_narrow_sigma_charge_loss')['unfloored_sum'].item()
    assert abs(raw - 1000.0) / 1000.0 > 0.10
