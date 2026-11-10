"""Keep notebook 08's model selection through every analysis configuration."""
from dataclasses import asdict
import json
from pathlib import Path
import runpy

import nbformat
import numpy as np
import pandas as pd
import pytest

from poodemo.data import Run, save_json
from poodemo import practical
from poodemo import diagnostics, ensemble_bundle, member_filter, pipeline

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO/'notebooks/08_reference_ratio_histograms.ipynb'


def test_config_copy_preserves_nested_routing_and_original_settings(tmp_path):
    base = Run(tmp_path/'run', {'reference_ratio_eta_floor': .2, 'bin_counts': [36]}, object())
    bundle = ensemble_bundle.BundleEvaluationRun(
        base, tmp_path/'bundle', {'name': 'unity100', 'campaign_id': 'campaign'})
    selected = member_filter.FilteredEvaluationRun(
        bundle, bundle.evaluation_root/'filtered/filter', {'identity': 'filter'})
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    for original in (base, bundle, selected):
        changed = original.with_config(reference_ratio_eta_floor=0., bin_counts=[4, 36])
        assert type(changed) is type(original)
        for parts in [('models', 'ratios', 'B_over_S'), ('selected', 'S_calibration.npy'),
                      ('quadrature', 'x.npy'), ('results', 'preselection.json'),
                      ('results', 'density_validation'), ('results', 'unbinned_scans.csv'),
                      ('plots', '08_test.pdf'), ('workspaces', 'learned')]:
            assert changed.path(*parts) == original.path(*parts)
        old, new = original, changed
        while True:
            assert old is not new and old.config is not new.config
            assert old.config == {'reference_ratio_eta_floor': .2, 'bin_counts': [36]}
            assert new.config == {'reference_ratio_eta_floor': 0., 'bin_counts': [4, 36]}
            assert old.model is new.model
            if not hasattr(old, 'base_run'):
                break
            old, new = old.base_run, new.base_run
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.fixture(params=['saved', 'migrated'])
def cells(request, monkeypatch):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    if request.param == 'migrated':
        monkeypatch.syspath_prepend(str(REPO/'scripts'))
        builder = runpy.run_path(str(REPO/'scripts/build_reference_notebooks.py'))
        # Exercise migration from the prior notebook, not just an existing cell.
        notebook.cells = [c for c in notebook.cells if c.id not in
                          ('reference-model-source', 'reference-model-intro')]
        notebook.cells = builder['configure_reference_cells'](notebook.cells, 8)
    nbformat.validate(notebook)
    return [c.source for c in notebook.cells if c.cell_type == 'code']


@pytest.mark.parametrize('ratio_source', ['unity', 'drive'])
@pytest.mark.parametrize('baseline_floor', [0., .2])
def test_notebook_studies_use_the_selected_models(cells, ratio_source, tmp_path, monkeypatch, baseline_floor):
    """Execute actual notebook cells, stubbing only expensive model/study calls."""
    base = Run(tmp_path/'run', {'reference_ratio_eta_floor': baseline_floor, 'bin_counts': [36]}, object())
    bundle_dir = tmp_path/'bundle'
    save_json(bundle_dir/'bundle.json', {})
    monkeypatch.setenv('POODEMO_UNITY_MODEL_DIR', str(bundle_dir))
    events = []

    def use_bundle(source, directory):
        assert source.root == base.root and directory == bundle_dir
        events.append('bundle')
        return ensemble_bundle.BundleEvaluationRun(
            source, directory, {'name': 'unity100', 'campaign_id': 'campaign'})

    def screen(source, *, policy):
        assert asdict(policy) == asdict(member_filter.MemberFilterPolicy())
        assert isinstance(source, ensemble_bundle.BundleEvaluationRun) == (ratio_source == 'unity')
        events.append('filter')
        evaluation = getattr(source, 'evaluation_root', base.root/'evaluations')/'filtered/filter'
        selected = member_filter.FilteredEvaluationRun(source, evaluation,
            {'identity': 'filter', 'inputs': {'policy': asdict(policy)}})
        report = selected.path('results', 'ensemble_filter', 'summary.csv')
        report.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'retained_members': [100, 100, 100, 96, 96]}).to_csv(report, index=False)
        return selected

    monkeypatch.setattr(ensemble_bundle, 'use_bundle', use_bundle)
    monkeypatch.setattr(practical, 'reuse_filtered_ensembles', screen)
    ns = {'run': base, 'ROOT': base.root, 'pd': pd, 'np': np, 'display': lambda x: None,
          'N_BINS': 36, 'REFERENCE_RATIO_ETA_MIN': baseline_floor}
    source = next(c for c in cells if 'run = filter_ratio_ensembles(' in c)
    source = source.replace('RATIO_SOURCE = "unity"', f'RATIO_SOURCE = "{ratio_source}"')
    exec(source, ns)
    selected = ns['run']
    # Rerunning the selection cell must not nest evaluations or change ROOT.
    exec(source, ns)
    assert ns['run'].evaluation_root == selected.evaluation_root
    assert ns['ROOT'] == base.root
    expected = ['bundle', 'filter'] if ratio_source == 'unity' else ['filter']
    assert events == expected * 2
    provenance = json.loads(selected.path('results', 'reference_ratio_model_source.json').read_text())
    assert provenance['ratio_source'] == ratio_source and provenance['filter_identity'] == 'filter'

    def check(view):
        assert isinstance(view, practical.PracticalRun)
        assert isinstance(view.base_run, member_filter.FilteredEvaluationRun)
        assert view.path('models', 'ratios') == selected.path('models', 'ratios')
        assert view.path('results') == selected.path('results')
        assert view.path('quadrature') == base.path('quadrature')

    def unbinned(view):
        check(view)
        events.append('unbinned')
        # This sentinel must be read from the selected evaluation by the sweep.
        pd.DataFrame({'q': [3.14]}).to_csv(view.path('results', 'unbinned_scans.csv'), index=False)
        return pd.DataFrame()

    def binning(view, **kwargs):
        check(view)
        events.append('binning')
        assert view.config['bin_counts'] == list(range(4, 61, 4))
        assert pipeline._saved_unbinned(view)['q'].item() == 3.14
        return {'scans': pd.DataFrame(), 'fidelity': pd.DataFrame()}

    def estimator(view, *, observable, n_bins, output_tag=None):
        check(view)
        assert observable == 'reference_ratio' and n_bins == 36
        assert view.config['bin_counts'] == [36]
        floor = view.config['reference_ratio_eta_floor']
        assert floor == ({'reference_ratio_estimator_eta_no_floor': 0.,
                          'reference_ratio_estimator_eta_floor_control': .2}.get(output_tag, baseline_floor))
        events.append(('estimator', floor))
        return pd.DataFrame({'eta': [.02, .02], 'model': ['direct reference_ratio histogram',
                            'analytic unbinned'], 'fit_valid': [True, True], 'width_valid': [True, True]})

    monkeypatch.setattr(practical, 'refresh_analytic_reference', unbinned)
    monkeypatch.setattr(pipeline, 'run_binning_study', binning)
    monkeypatch.setattr(diagnostics, 'run_estimator_study', estimator)
    for code in cells:
        if any(marker in code for marker in ('unbinned = refresh_analytic_reference(run)',
                                             'binning = run_binning_study(',
                                             'estimator_study = run_estimator_study(')):
            exec(code, ns)
    other_floor = .2 if baseline_floor == 0. else 0.
    assert events[-4:] == ['unbinned', 'binning', ('estimator', baseline_floor), ('estimator', other_floor)]
    assert ns['run'].config['reference_ratio_eta_floor'] == baseline_floor
    assert selected.path('results', 'reference_ratio_estimator_eta_floor_comparison.csv').is_file()
    assert not base.path('results').exists()


def test_missing_bundle_cannot_fall_back_to_original_models(cells, tmp_path, monkeypatch):
    monkeypatch.setenv('POODEMO_UNITY_MODEL_DIR', str(tmp_path/'missing'))
    def unexpected(*args, **kwargs):
        pytest.fail('Do not load, filter, download, or train after a missing bundle')
    monkeypatch.setattr(ensemble_bundle, 'use_bundle', unexpected)
    monkeypatch.setattr(ensemble_bundle, 'fetch_bundle', unexpected)
    monkeypatch.setattr(member_filter, 'filter_ratio_ensembles', unexpected)
    source = next(c for c in cells if 'run = filter_ratio_ensembles(' in c)
    ns = {'run': Run(tmp_path/'run', {}, None)}
    with pytest.raises(FileNotFoundError, match='Copy the complete Unity model bundle'):
        exec(source, ns)


def test_all_outputs_follow_the_selected_run(cells):
    for source in cells:
        assert 'ROOT / "results"' not in source
        assert 'ROOT / "plots"' not in source
        assert 'replace(run,' not in source


def test_migration_is_repeatable_and_preserves_user_controls(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'scripts'))
    builder = runpy.run_path(str(REPO/'scripts/build_reference_notebooks.py'))
    migrate = builder['configure_reference_cells']
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    setup = next(c for c in notebook.cells if 'N_BINS = 36' in c.source)
    setup.source = setup.source.replace('N_BINS = 36', 'N_BINS = 48')
    selection = next(c for c in notebook.cells if c.id == 'reference-model-source')
    selection.source = selection.source.replace('RATIO_SOURCE = "unity"', 'RATIO_SOURCE = "drive"')
    first = [c.source for c in migrate(notebook.cells, 8)]
    assert first == [c.source for c in migrate(notebook.cells, 8)]
    assert any('N_BINS = 48' in c for c in first)
    assert any('RATIO_SOURCE = "drive"' in c for c in first)
