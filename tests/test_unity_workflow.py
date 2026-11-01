"""Unity data identity, complete-member publication and bounded ensemble inference."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from poodemo.data import Run, SAMPLES, file_digest, save_json
from poodemo.ensemble_bundle import fetch_bundle, use_bundle, verify_bundle
from poodemo.physics import PhysicsModel
from poodemo.training import Predictor, StreamingRatioEnsemble
from unity import workflow as w
from unity.submit import commands


def small_config():
    cfg = w.read(w.REPO / 'unity/config.json')
    cfg.update(name='unity-smoke', members=2, generation_chunk=1024,
               selected_counts={'train': 64, 'validation': 32, 'calibration': 32})
    cfg['training'].update(epochs=2, width=8, depth=2, batch_size=16, accelerator='cpu')
    return cfg


class SelectAll:
    def predict_proba(self, x, batch_size=8192):
        return np.tile([1., 0., 0.], (len(x), 1))


def test_frozen_drive_identity_and_campaign_change_detection(tmp_path):
    identity = w.source_identity(w.SOURCE)
    assert identity['selection']['threshold'] == 0.28696612960213425
    assert identity['selector_sha256'] == w.read(w.SOURCE / 'provenance.json')['files']['preselection/model.pt']['sha256']
    cfg = small_config()
    m = w.initialize(tmp_path, cfg)
    assert w.initialize(tmp_path, cfg) == m
    changed = {**cfg, 'members': 3}
    with pytest.raises(ValueError, match='changed'):
        w.initialize(tmp_path, changed)
    state = tmp_path / 'source/selection_state.json'
    record = w.read(state); record['threshold'] = .1; save_json(state, record)
    with pytest.raises(ValueError, match='do not match'):
        w.campaign(tmp_path)


def test_fixed_selected_counts_split_independence_resume_and_corruption(tmp_path):
    cfg = small_config()
    m = w.initialize(tmp_path, cfg)
    w.generate(tmp_path, 'S', SelectAll())
    arrays = {split: w.checked_array(tmp_path, 'S', split, m) for split in cfg['selected_counts']}
    for split, x in arrays.items():
        assert x.shape == (cfg['selected_counts'][split], 3)
        assert x.dtype == np.float32
    rows = [tuple(row) for x in arrays.values() for row in x]
    assert len(rows) == len(set(rows))
    path = w.selected_path(tmp_path, 'S', 'train')
    before = file_digest(path)
    w.generate(tmp_path, 'S', SelectAll())
    assert file_digest(path) == before
    # Incomplete split restarts from its own original stream, independently of others.
    path.unlink(); path.with_suffix('.json').unlink()
    w.generate(tmp_path, 'S', SelectAll())
    assert file_digest(path) == before
    with path.open('r+b') as stream:
        stream.seek(-4, 2); stream.write(b'xxxx')
    with pytest.raises(ValueError, match='provenance'):
        w.checked_array(tmp_path, 'S', 'train', m)


def test_five_hundred_unique_jobs_and_submission_resources(tmp_path):
    cfg = w.read(w.REPO / 'unity/config.json')
    w.initialize(tmp_path, cfg)
    seeds = set()
    for index in range(500):
        ti, mi, pair = w.task_member(index, 100)
        seeds.add(cfg['seed'] + 103*ti + 100003*mi)
        assert pair == w.RATIO_TASKS[ti]
    assert len(seeds) == 500
    with pytest.raises(ValueError):
        w.task_member(500, 100)
    jobs = list(commands(tmp_path, Path('/project/env/bin/python'), 'all', 20,
                         'pi_rclsa_umass_edu', 'gpu', 'cpu', '1-00:00:00', ''))
    assert '--array=0-5%6' in jobs[0][1]
    assert '--gpus=1' not in jobs[0][1]
    assert '--array=0-499%20' in jobs[1][1]
    assert '--gpus=1' in jobs[1][1]
    assert '--dependency=afterok:<generate-job-id>' in jobs[1][1]
    assert '--dependency=afterok:<train-job-id>' in jobs[2][1]
    assert f'--chdir={tmp_path}' in jobs[1][1]


def test_streaming_mean_is_arithmetic_ratio_mean_and_reads_each_model_once(tmp_path, monkeypatch):
    save_json(tmp_path/'ensemble.json', {'members': ['a', 'b'], 'labels': ['S', 'B'],
              'streaming': True, 'log_normalization': np.log(3.)})
    calls = []
    class Known:
        def __init__(self, value): self.value = value
        def predict_raw_log_ratio(self, x, batch_size): return np.full(len(x), np.log(self.value))
        def predict_log_ratio(self, x, batch_size): return np.full(len(x), np.log(self.value / 2))
    def load(path, device):
        calls.append(path.name)
        return Known(2. if path.name == 'a' else 8.)
    monkeypatch.setattr(Predictor, 'load', load)
    predictor = StreamingRatioEnsemble(tmp_path)
    x = np.zeros((37, 3))
    np.testing.assert_allclose(predictor.predict_raw_ratio(x), 5.)
    assert calls == ['a', 'b']
    calls.clear()
    np.testing.assert_allclose(predictor.predict_ratio(x), 2.5 / 3.)
    assert calls == ['a', 'b']
    np.testing.assert_allclose(predictor.predict_ratio(x, normalized=False), 2.5)


HAS_TRAINING = all(importlib.util.find_spec(n) for n in ('torch', 'pytorch_lightning', 'nsbi_common_utils'))


@pytest.mark.skipif(not HAS_TRAINING, reason='optional training dependencies')
def test_tiny_real_campaign_export_reload_and_drive_evaluation(tmp_path, monkeypatch):
    import torch
    torch.set_num_threads(1)
    root = tmp_path / 'work'
    cfg = small_config()
    w.initialize(root, cfg)
    for sample in SAMPLES:
        w.generate(root, sample)  # Real frozen Drive preselection and analytic generator.
    with pytest.raises(FileNotFoundError):
        w.collect(root, tmp_path / 'incomplete')
    assert not (tmp_path / 'incomplete').exists()
    for index in range(10):
        w.train(root, index)
    # Completed jobs are safely reusable, with the same checkpoint contents.
    first = root/'members/SBI_over_S/member_000/model.pt'
    before = file_digest(first)
    w.train(root, 0)
    assert file_digest(first) == before
    for ti in range(5):
        w.normalize(root, ti)
    bundle = w.collect(root, tmp_path/'bundle')
    manifest = verify_bundle(bundle)
    assert manifest['members_per_task'] == 2
    files = list(bundle.rglob('*'))
    assert len(list(bundle.rglob('model.pt'))) == 10
    assert not any(p.suffix in ('.npy', '.npz', '.ckpt') for p in files)
    assert not any(p.name == 'checkpoints' for p in files)
    calibration = np.load(w.selected_path(root, 'S', 'calibration'))
    ensemble = w.load_predictor(bundle/'ratios/SBI_over_S', device='cpu')
    np.testing.assert_allclose(ensemble.predict_ratio(calibration).mean(), 1., atol=1e-7)
    base = tmp_path/'drive'
    shutil.copytree(w.SOURCE/'preselection', base/'models/preselection')
    save_json(base/'selected/selection_state.json', manifest['identity']['selection'])
    save_json(base/'results/preselection.json', w.read(w.SOURCE/'preselection.json'))
    physics = PhysicsModel.from_dict(manifest['identity']['physics'])
    run = Run(base, {}, physics)
    evaluation = use_bundle(run, bundle)
    assert evaluation.path('raw', 'S.npy') == base/'raw/S.npy'
    assert evaluation.path('quadrature', 'x.npy') == base/'quadrature/x.npy'
    assert evaluation.path('results', 'density_validation') == base/'results/density_validation'
    assert evaluation.path('models', 'ratios', 'SBI_over_S') == bundle/'ratios/SBI_over_S'
    assert evaluation.path('results', 'unbinned_scans.csv').is_relative_to(base/'evaluations')
    with pytest.raises(ValueError, match='different physics'):
        use_bundle(replace(run, model=replace(physics, phase_cos=-.1)), bundle)
    # Exercise actual Git LFS pointers, collection commits, sparse download and
    # model materialization against a local repository, without GitHub writes.
    if subprocess.run(['git', 'lfs', 'version'], capture_output=True).returncode == 0:
        from unity import publish as publisher
        repo = tmp_path/'publication-source'
        repo.mkdir()
        shutil.copy2(w.REPO/'.gitattributes', repo/'.gitattributes')
        for args in [('init', '-b', 'main'), ('config', 'user.name', 'Test'),
                     ('config', 'user.email', 'test@example.invalid'), ('add', '.gitattributes'),
                     ('commit', '-m', 'Initial LFS attributes')]:
            subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)
        monkeypatch.setattr(publisher, 'REPO', repo)
        checkout = tmp_path/'publication'
        sha = publisher.publish(root, checkout)
        pointer = subprocess.check_output(['git', '-C', str(checkout), 'show',
            'HEAD:unity/ensembles/unity-smoke/ratios/SBI_over_S/member_000/model.pt'], text=True)
        assert pointer.startswith('version https://git-lfs.github.com/spec/v1')
        download = fetch_bundle('unity-smoke', sha, tmp_path/'cache', repository=str(checkout))
        assert verify_bundle(download)['campaign_id'] == manifest['campaign_id']
        restored = w.load_predictor(download/'ratios/SBI_over_S', device='cpu')
        np.testing.assert_array_equal(restored.predict_ratio(calibration), ensemble.predict_ratio(calibration))
    (bundle/'ratios/SBI_over_S/member_000/model.pt').write_bytes(b'broken')
    with pytest.raises(ValueError, match='checksum'):
        verify_bundle(bundle)
