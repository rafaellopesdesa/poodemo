"""Notebook 03 loads a transferred directory without network access or training."""
import ast
import json
from pathlib import Path
import runpy
import shutil

import pytest

from poodemo.data import Run, file_digest, save_json
from poodemo import ensemble_bundle
from poodemo.physics import PhysicsModel
from poodemo.pipeline import RATIO_TASKS
from unity.workflow import source_identity

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(params=['saved', 'generated'])
def notebook_code(request, monkeypatch):
    if request.param == 'saved':
        notebook = json.loads((REPO/'notebooks/03_unbinned_nsbi.ipynb').read_text())
        cells = notebook['cells']
    else:
        builder = runpy.run_path(str(REPO/'scripts/build_notebooks.py'))
        notebooks = {}
        monkeypatch.setitem(builder['build'].__globals__, 'write',
                            lambda name, cells: notebooks.update({name: cells}))
        builder['build']()
        cells = notebooks['03_unbinned_nsbi.ipynb']
    return [''.join(cell['source']).strip() for cell in cells if cell['cell_type'] == 'code']


@pytest.mark.parametrize('ratio_source', ['unity', 'drive'])
def test_top_to_bottom_model_selection_precedes_validation(
        notebook_code, ratio_source, tmp_path, monkeypatch):
    """Execute the real setup/diagnostic cells in order without costly predictions."""
    import os
    import numpy as np
    import pandas as pd
    from poodemo import closure_validation, pipeline, member_filter

    events = []
    run = Run(tmp_path/'run', {}, PhysicsModel())
    bundle = tmp_path/'models'
    save_json(bundle/'bundle.json', {})
    monkeypatch.setenv('POODEMO_UNITY_MODEL_DIR', str(bundle))
    samples = {'S': np.zeros((2, 3))}

    def load_models(base_run, directory):
        assert base_run is run
        assert directory == bundle
        events.append('load')
        return ensemble_bundle.BundleEvaluationRun(
            run, bundle, {'name': 'test', 'campaign_id': 'notebook-order'})

    def check_run(selected_run):
        if ratio_source == 'unity':
            assert isinstance(selected_run, ensemble_bundle.BundleEvaluationRun)
            assert selected_run.path('models', 'ratios') == bundle/'ratios'
        else:
            assert selected_run is run

    def prepare_bank(selected_run, **kwargs):
        check_run(selected_run)
        events.append('bank')
        return {'samples': samples, 'generated_counts': {'S': 2}}

    def validate(selected_run, validation_samples, **kwargs):
        assert isinstance(selected_run, member_filter.FilteredEvaluationRun)
        check_run(selected_run.base_run)
        assert validation_samples is samples
        assert kwargs['validate_members'] is False
        events.append('validate')
        return pd.DataFrame()

    def train(selected_run, **kwargs):
        assert ratio_source == 'drive', 'Unity mode must never train in Colab'
        check_run(selected_run)
        assert 'validation_samples' not in kwargs  # Diagnostics follow filtering.
        events.append('train')
        return pd.DataFrame()

    def filter_members(selected_run, *, policy):
        check_run(selected_run)
        assert isinstance(policy, member_filter.MemberFilterPolicy)
        assert policy.catastrophic_peer_factor == 100.
        events.append('filter')
        filtered = member_filter.FilteredEvaluationRun(selected_run, tmp_path/'filtered', {})
        report = filtered.path('results', 'ensemble_filter', 'summary.csv')
        report.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'retained_members': [4]}).to_csv(report, index=False)
        return filtered

    monkeypatch.setattr(ensemble_bundle, 'use_bundle', load_models)
    monkeypatch.setattr(closure_validation, 'prepare_validation_bank', prepare_bank)
    monkeypatch.setattr(ensemble_bundle, 'validate_bundle_ratios', validate)
    monkeypatch.setattr(member_filter, 'prepare_ratio_members', train)
    monkeypatch.setattr(member_filter, 'filter_ratio_ensembles', filter_members)
    namespace = {'os': os, 'Path': Path, 'run': run, 'MODE': 'smoke',
                 'np': np, 'pd': pd, 'display': lambda value: None}
    for source in notebook_code:
        if 'use_bundle(source_run,' in source:
            source = source.replace('RATIO_SOURCE = "unity"', f'RATIO_SOURCE = "{ratio_source}"')
        if any(call in source for call in ('use_bundle(source_run,',
                                          'prepare_validation_bank(run,',
                                          'ratio_diagnostics = validate_bundle_ratios(')):
            exec(source, namespace)
    assert events == (['load', 'bank', 'filter', 'validate'] if ratio_source == 'unity'
                      else ['bank', 'train', 'filter', 'validate'])
    assert namespace['ROOT'] == namespace['run'].evaluation_root


def source_constants():
    tree = ast.parse((REPO/'scripts/build_notebooks.py').read_text())
    return {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ('UNITY_SOURCE_CODE', 'UNITY_SOURCE_INTRO')}


def test_notebook_source_matches_builder_and_default_drive_path():
    constants = source_constants()
    notebook = json.loads((REPO/'notebooks/03_unbinned_nsbi.ipynb').read_text())
    for cell_id, key in [('unity-source-code', 'UNITY_SOURCE_CODE'),
                         ('unity-source-intro', 'UNITY_SOURCE_INTRO')]:
        cell = next(c for c in notebook['cells'] if c['id'] == cell_id)
        assert ''.join(cell['source']).strip() == constants[key].strip()
    code = constants['UNITY_SOURCE_CODE']
    assert '/content/drive/MyDrive/poodemo/unity_models/paper-distinct-s-v4-unity100' in code
    assert 'fetch_bundle' not in code
    assert 'UNITY_REF' not in code
    assert 'RATIO_SOURCE = "unity"' in code
    ast.parse(code)


def test_missing_transfer_stops_before_model_loading(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('No downloading, model loading or training should occur')
    monkeypatch.setattr(ensemble_bundle, 'fetch_bundle', unexpected)
    monkeypatch.setattr(ensemble_bundle, 'use_bundle', unexpected)
    monkeypatch.setenv('POODEMO_UNITY_MODEL_DIR', str(tmp_path/'missing'))
    import os
    run = Run(tmp_path/'run', {}, PhysicsModel())
    namespace = {'os': os, 'Path': Path, 'run': run}
    with pytest.raises(FileNotFoundError, match='Copy the complete Unity model bundle'):
        exec(source_constants()['UNITY_SOURCE_CODE'], namespace)
    assert namespace['run'] is run


def test_transferred_directory_verified_without_download(tmp_path, monkeypatch):
    import os
    from poodemo import pipeline
    def unexpected(*args, **kwargs):
        raise AssertionError('Transferred models must not download or train')
    monkeypatch.setattr(ensemble_bundle, 'fetch_bundle', unexpected)
    monkeypatch.setattr(pipeline, 'train_ratios', unexpected)
    source = REPO/'unity/source'
    identity = source_identity(source)
    run_root = tmp_path/'drive/poodemo/runs/paper-distinct-s-v4'
    shutil.copytree(source/'preselection', run_root/'models/preselection')
    save_json(run_root/'selected/selection_state.json', identity['selection'])
    save_json(run_root/'results/preselection.json', json.loads((source/'preselection.json').read_text()))
    run = Run(run_root, {}, PhysicsModel.from_dict(identity['physics']))
    bundle = tmp_path/'drive/poodemo/unity_models/transferred'
    # Model bytes are only hashed during selection, not deserialized or evaluated.
    for num, den in RATIO_TASKS:
        task = bundle/'ratios'/f'{num}_over_{den}'
        (task/'member_000').mkdir(parents=True)
        (task/'member_000/model.pt').write_bytes(b'test-inference-model')
        save_json(task/'ensemble.json', {'members': ['member_000'], 'labels': [den, num],
                                       'streaming': True, 'log_normalization': 0.})
    save_json(bundle/'campaign.json', {'campaign_id': 'directory-test'})
    hashes = {str(p.relative_to(bundle)): file_digest(p) for p in bundle.rglob('*') if p.is_file()}
    save_json(bundle/'bundle.json', {'schema': 1, 'name': 'transferred', 'campaign_id': 'directory-test',
              'identity': identity, 'members_per_task': 1, 'files': hashes})
    monkeypatch.setenv('POODEMO_UNITY_MODEL_DIR', str(bundle))
    namespace = {'os': os, 'Path': Path, 'run': run}
    code = source_constants()['UNITY_SOURCE_CODE']
    exec(code, namespace)
    loaded = namespace['run']
    assert loaded.bundle == bundle.resolve()
    assert loaded.base_run is run
    assert loaded.path('raw', 'S.npy') == run.path('raw', 'S.npy')
    assert loaded.path('quadrature', 'x.npy') == run.path('quadrature', 'x.npy')
    assert namespace['ROOT'].is_relative_to(run_root/'evaluations')
    exec(code, namespace)  # Re-running the source cell retains the original Drive run.
    assert namespace['run'].base_run is run
    (bundle/'ratios/SBI_over_S/member_000/model.pt').write_bytes(b'incomplete-transfer')
    with pytest.raises(ValueError, match='checksum mismatch'):
        exec(code, namespace)
