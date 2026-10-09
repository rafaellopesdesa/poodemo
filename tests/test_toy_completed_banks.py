"""Section 7 reads completed section-6 banks without extending or mutating them."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import poodemo.toy_convergence_banks as module
from test_toy_convergence_banks import _context, _original


def _tree_hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


def _same_bank(a, b):
    assert a['metadata'] == b['metadata']
    assert a['banks'].keys() == b['banks'].keys()
    for key, bank in a['banks'].items():
        other = b['banks'][key]
        assert bank.keys() == other.keys()
        assert bank['metadata'] == other['metadata']
        for name in ('rates', 'rate_covariance'):
            np.testing.assert_array_equal(bank[name], other[name])
        for name in ('templates', 'component_covariances', 'edges'):
            assert bank[name].keys() == other[name].keys()
            for prediction, value in bank[name].items():
                np.testing.assert_array_equal(value, other[name][prediction])


def _fail(*args, **kwargs):
    raise AssertionError('The completed-bank loader must not write or generate anything')


@pytest.mark.parametrize('bins', [(12, 36), (36,)])
def test_completed_bank_loader_is_exact_and_read_only(tmp_path, monkeypatch, bins):
    context = _context(tmp_path)
    _original(context)
    expected = module.prepare_convergence_banks(context, extra_multiples=(10,), bin_counts=bins)
    before = _tree_hashes(tmp_path)
    # The loader must not need a simulator, selector or network, even when the
    # manifest includes extended prefixes that prepare() generated previously.
    for name in ('run', 'selector', 'workspace'):
        context.pop(name)
    monkeypatch.setattr(module, '_atomic_json', _fail)
    monkeypatch.setattr(module, '_atomic_npz', _fail)
    monkeypatch.setattr(Path, 'mkdir', _fail)
    actual = module.load_completed_convergence_banks(context)
    _same_bank(expected, actual)
    assert _tree_hashes(tmp_path) == before


def test_missing_manifest_fails_without_creating_output_directory(tmp_path):
    output = tmp_path / 'does_not_exist'
    context = _context(output)
    with pytest.raises(ValueError, match='complete notebook 12 section 6'):
        module.load_completed_convergence_banks(context)
    assert not output.exists()


def test_missing_completed_checkpoint_does_not_resume_work(tmp_path):
    context = _context(tmp_path)
    _original(context)
    module.prepare_convergence_banks(context)
    directory = tmp_path / 'convergence' / 'bank_convergence' / 'integration'
    (directory / 'replica_1_NI_800.npz').unlink()
    assert (directory / 'replica_1_NI_work.npz').exists()
    before = _tree_hashes(tmp_path)
    with pytest.raises(ValueError, match='complete notebook 12 section 6'):
        module.load_completed_convergence_banks(context)
    assert _tree_hashes(tmp_path) == before


def test_loader_rejects_replaced_original_checkpoint(tmp_path):
    context = _context(tmp_path)
    _original(context)
    module.prepare_convergence_banks(context)
    checkpoint = tmp_path / 'integration' / 'replica_1_NI_160.npz'
    with np.load(checkpoint, allow_pickle=False) as saved:
        fields = {key: saved[key].copy() for key in saved.files}
    fields['rng_state'] = np.array(json.dumps(np.random.default_rng(8).bit_generator.state))
    np.savez_compressed(checkpoint, **fields)
    before = _tree_hashes(tmp_path)
    with pytest.raises(ValueError, match='snapshots changed'):
        module.load_completed_convergence_banks(context)
    assert _tree_hashes(tmp_path) == before


def test_loader_rejects_wrong_fingerprint_in_extension(tmp_path):
    context = _context(tmp_path)
    _original(context)
    module.prepare_convergence_banks(context)
    checkpoint = tmp_path / 'convergence' / 'bank_convergence' / 'integration' / 'replica_1_NI_800.npz'
    with np.load(checkpoint, allow_pickle=False) as saved:
        fields = {key: saved[key].copy() for key in saved.files}
    fields['fingerprint'] = np.array('different')
    np.savez_compressed(checkpoint, **fields)
    with pytest.raises(ValueError, match='different fingerprint'):
        module.load_completed_convergence_banks(context)
