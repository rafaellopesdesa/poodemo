"""10M cap, streamed direct templates, independent-source errors and 08 closure."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import binom

from poodemo import colab_discriminators as module, pipeline
from poodemo.colab_mc import fill_colab_histograms, convergence_scans
from poodemo.colab_reference import publish_08_baseline, load_08_baseline
from poodemo.data import SAMPLES, PROCESSES
from poodemo.discriminator_precision import save_precision_tables, plot_mc_precision
from poodemo.inference import component_coefficients
from poodemo.practical import refresh_analytic_reference
from test_colab_mc import setup, prepare


def test_fixed_sample_variance_is_unbiased_and_signed_sources_are_independent():
    # Enumerate all possible counts from a known binomial experiment, not a
    # duplicate of the variance implementation.
    n, probability, rate = 7, .13, 42.
    k = np.arange(n+1)
    mean, variance = module.source_moments(k[None], [rate], n)
    probabilities = binom.pmf(k, n, probability)
    np.testing.assert_allclose(variance @ probabilities, rate**2*probability*(1-probability)/n)
    np.testing.assert_allclose(mean @ probabilities, rate*probability)
    counts = np.array([[4, 0, 0], [0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    rates = np.ones(6)*10
    result = module.precision_table(counts, rates, 10, mu=.25)
    # -.25*S + .5*SBI + .5*B + NI: cancellation yields -0.5 in bin 0.
    assert result.raw_yield.iloc[0] == pytest.approx(-.5)
    assert result.mc_sigma.iloc[0] > 0 and np.isnan(result.relative_mc_sigma.iloc[0])
    assert list(result.status) == ['nonpositive', 'positive', 'no_sampled_support']
    expected_var = .25**2*(100*.4*.6/9) + .5**2*(100*.1*.9/9)
    assert result.mc_sigma.iloc[0]**2 == pytest.approx(expected_var)
    # A sampled-empty component has undefined relative precision, not 0%.
    assert module.precision_table(counts, rates, 10, source='SBI').relative_mc_sigma.isna().all()


def test_stream_matches_independent_numpy_histogram_and_resumes(setup, monkeypatch):
    run, calls = setup
    samples = prepare(run)
    before = calls.copy()
    read = module._read_array
    calls_read = []
    def interrupted(*args):
        calls_read.append(True)
        if len(calls_read) == 5:
            raise RuntimeError('interruption')
        return read(*args)
    monkeypatch.setattr(module, '_read_array', interrupted)
    with pytest.raises(RuntimeError, match='interruption'):
        module.fixed_counts(samples, 4, progress=lambda _:None)
    monkeypatch.setattr(module, '_read_array', read)
    result = module.fixed_counts(samples, 4, progress=lambda _:None)
    for i, state in enumerate(samples['manifests'].values()):
        values = []
        for name in module.AXES:
            raw = np.concatenate([read(samples['directory'], chunk['ratios'][name]) for chunk in state['chunks']])[:samples['target']]
            j = PROCESSES.index(name)
            values.append(expit(result['powers'][name]*np.log(raw/samples['normalizers'][j]*samples['normalizers'][0])))
        expected = np.histogramdd(np.column_stack(values), bins=[result['edges']]*3)[0]
        np.testing.assert_array_equal(result['counts'][i], expected)
    monkeypatch.setattr(module, '_read_array', lambda *a:pytest.fail('Completed counts should not reread ratios'))
    np.testing.assert_array_equal(module.fixed_counts(samples, 4, progress=lambda _:None)['counts'], result['counts'])
    assert calls == before  # Neither generation nor NN inference occurred.
    with pytest.raises(ValueError, match='10M'):
        module.fixed_counts({**samples, 'target':10_000_001}, 4)


def test_08_to_10_direct_closure_precision_and_actual_notebook_cells(setup, tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from poodemo import practical
    from poodemo import discriminator_unbinned as unbinned
    from poodemo.discriminator_study import plot_discriminator_study
    base, calls = setup
    base = base.with_config(mc_zero_max_bin=.01, mc_zero_max_total=.05)
    samples = prepare(base)
    grid = pipeline._positive_study_grid(base)
    bank = fill_colab_histograms(base, samples, etas=grid, bin_counts=[4], prefixes=[13], progress=lambda _:None)
    run = bank.study_run(base)
    # Only stub the separate unbinned oracle; the MC histogram path is real.
    nominal = np.broadcast_to(samples['rates'][:4, None], (4, 10)).copy()
    aq = dict(nominal=nominal, down=nominal[None].copy(), up=nominal[None].copy(),
              weights=np.full(10, .1), truth=nominal[1]+nominal[3])
    monkeypatch.setattr(pipeline, 'prepare_quadrature', lambda _:aq)
    monkeypatch.setattr(module, 'cached_analytic_quadrature', lambda _:aq)
    refresh_analytic_reference(run)
    learned = pd.read_csv(run.path('results', 'unbinned_scans.csv'))
    learned['model'] = unbinned.LEARNED_LABEL
    owner = base.base_run
    owner.path('results').mkdir(parents=True, exist_ok=True)
    learned.to_csv(owner.path('results', 'unbinned_scans.csv'), index=False)
    pd.DataFrame([dict(model=unbinned.LEARNED_LABEL, systematics=s, nll_at_asimov=0.)
                  for s in (False, True)]).to_csv(owner.path('results', 'unbinned_fits.csv'), index=False)
    monkeypatch.setattr(unbinned, '_load_objective', lambda _:(lambda p:50*(p[0]-1)**2+10*p[1]**2, []))
    convergence_scans(run, bank, etas=grid, n_bins=4)
    marker = publish_08_baseline(base, run, samples, bank, 4)
    with pytest.raises(ValueError, match='selected-event limit'):
        load_08_baseline(base, 4, max_selected=12)
    run, cached = load_08_baseline(base, 4, max_selected=10_000_000, local_cache=tmp_path/'stage', progress=lambda _:None)
    before = calls.copy()
    monkeypatch.setattr(practical, 'prepare_binning_bank', lambda *a:pytest.fail('Old dense bank requested'))
    sentinel = run.path('results', 'reference_ratio_mc_convergence.csv').read_bytes()
    # Execute the notebook's actual study/plot cells on the completed fixture.
    notebook = json.loads((Path(__file__).resolve().parents[1]/'notebooks/10_discriminator_histograms.ipynb').read_text())
    cells = {c['id']:''.join(c['source']) for c in notebook['cells']}
    ns = dict(run=run, samples_08=cached, N_BINS=4, POWERS=module.DEFAULT_POWERS,
              RECAP_BIN_COUNTS=[4, 8, 12, 16],
              OUTPUT_TAG='test10', PRECISION_MUS=[.02, .122, 1.], SOURCE_3D_PLOTS=['NI_down'],
              display=lambda *a:None, plt=plt, pd=pd, np=np)
    for name in ['study', 'comparison-plots', 'recap-plots', 'precision-plots']:
        exec(cells[name], ns)
    result = ns['study']
    assert result['baseline_checks'].passed.all()
    assert result['config']['selected_per_source'] == 13
    assert set(result['resolution_scans'].n_bins.dropna()) == {4, 8, 12, 16}
    assert unbinned.LEARNED_LABEL in set(result['scans'].model)
    assert unbinned.LEARNED_LABEL in set(result['resolution_information'].model)
    assert result['config']['learned_unbinned']['filter_identity'] == 'fixed'
    assert result['config']['no_generation'] and not result['config']['adaptive_binning']
    assert calls == before
    assert sentinel == run.path('results', 'reference_ratio_mc_convergence.csv').read_bytes()
    for d in [1, 2, 3]:
        data = pd.read_csv(result['precision_output']/f'fixed_{d}d_total.csv.gz')
        assert len(data) == 3*4**d
        assert np.isfinite(data.loc[data.status == 'positive', 'relative_mc_sigma']).all()
        source = pd.read_csv(result['precision_output']/f'fixed_{d}d_sources.csv.gz')
        assert len(source) == 6*4**d
    assert len(list(ns['FIGURE_DIR'].glob('10_mc_precision_*.png'))) == 7
    assert len(list(ns['FIGURE_DIR'].glob('10_mc_precision_*.pdf'))) == 7
    assert len(list(ns['FIGURE_DIR'].glob('10_recap_*.png'))) == 3
    precision = pd.read_csv(result['precision_output']/'moving_total.csv.gz')
    assert set(precision.n_bins) == {4, 8, 12, 16}
    # Corrupted direct counts must never be silently refilled/replaced.
    checkpoint = cached['histogram_directory']/'counts.npz'
    with np.load(checkpoint) as z:
        payload = {name:z[name] for name in z.files}
    payload['counts'][0, 0, 0, 0] += 1
    np.savez_compressed(checkpoint, **payload)
    with pytest.raises(ValueError, match='checkpoint changed'):
        module.completed_histograms(run, cached)


def test_learned_curvature_uses_actual_objective_and_checks_steps():
    from poodemo.discriminator_unbinned import objective_curvature
    matrix = np.array([[5., 2.], [2., 7.]])
    def objective(p):
        delta = p-np.array([1., 0.])
        return 3. + delta @ matrix @ delta + .2*delta[0]
    measured, error = objective_curvature(objective, 1.)
    np.testing.assert_allclose(measured, matrix, rtol=1e-7)
    assert error < 1e-6


def test_missing_reference_never_rebuilds_quadrature(tmp_path, monkeypatch):
    from types import SimpleNamespace
    run = SimpleNamespace(path=lambda *parts:tmp_path.joinpath(*parts))
    monkeypatch.setattr(pipeline, 'prepare_quadrature', lambda _:pytest.fail('Reference rebuild attempted'))
    with pytest.raises(FileNotFoundError, match='completed analytical quadrature'):
        module.cached_analytic_quadrature(run)


def test_saved_learned_workspace_rebases_after_drive_remount(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from poodemo import toolkit
    from poodemo.discriminator_unbinned import _load_objective
    array = tmp_path/'ratio.npy'
    np.save(array, [1., 2.])
    workspace = tmp_path/'workspace.json'
    workspace.write_text(json.dumps({'channels':[{'ratios':'/old/mount/ratio.npy'}]}))
    def load(spec):
        values = np.load(spec['channels'][0]['ratios'])
        return SimpleNamespace(model=lambda p:float(values @ p))
    monkeypatch.setattr(toolkit, 'load_model', load)
    objective, artifacts = _load_objective(workspace)
    assert objective([3., 4.]) == 11.
    assert artifacts[0]['file'] == 'ratio.npy' and len(artifacts[0]['sha256']) == 64
    array.unlink()
    with pytest.raises(FileNotFoundError, match='workspace array'):
        _load_objective(workspace)


def test_builder_is_current_and_notebook_has_no_generation_entry_point(monkeypatch):
    import nbformat
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    from build_comparison_notebooks import discriminator_cells
    built = discriminator_cells()
    actual = nbformat.read(root/'notebooks/10_discriminator_histograms.ipynb', as_version=4)
    assert [c.source for c in built] == [c.source for c in actual.cells]
    sources = '\n'.join(c.source for c in built if c.cell_type == 'code')
    for forbidden in ['prepare_colab_samples(', 'sample_component(', 'prepare_simulation_bank(', '250_000']:
        assert forbidden not in sources
    assert 'MAX_SELECTED_PER_SOURCE = 10_000_000' in sources
    assert all(not c.outputs and c.execution_count is None for c in actual.cells if c.cell_type == 'code')
